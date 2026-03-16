#include "mpa_ported_demo_renderer.h"

#include <GLES3/gl31.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log_manager.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PORTED_SHADOW_W (2048)
#define PORTED_SHADOW_H (2048)
#define PORTED_MAX_INSTANCES (128)
#define PORTED_MAX_PILLARS (128)
#define PORTED_GROUND_RADIUS (200.0f)
#define PORTED_ENABLE_SHADOWS (0)

typedef struct {
    GLuint vao;
    GLuint vbo;
    GLsizei vertexCount;
    GLuint diffuseTex;
    bool hasTexAttrib;
} PortedMesh;

typedef struct {
    bool initialized;

    GLuint mainProgram;
    GLuint shadowProgram;
    GLuint shadowInstProgram;
    GLuint instProgram;
    GLuint skyProgram;

    GLuint depthFbo;
    GLuint depthTex;

    PortedMesh cubeMesh;
    PortedMesh skyMesh;
    PortedMesh carMesh;
    PortedMesh parkedMesh;

    GLuint instModelBuf;

    float timeSec;
    float carRotation;
    float carSpeed;
    float steeringInput;

    float carPos[3];
    float smoothCamPos[3];

    bool usingFallbackCar;
    bool usingFallbackParked;
} PortedRenderer;

typedef struct { float x, y, z; } Vec3;
typedef struct { float m[16]; } Mat4;
typedef struct { float x, z; } Pillar;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t strideFloats;
    uint32_t vertexCount;
} MeshBinHeader;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
} TexBinHeader;

#define MESH_MAGIC (0x4D534831u)
#define TEX_MAGIC  (0x54455831u)

static PortedRenderer g_ported = {0};

static const char* kShadowVs =
"#version 310 es\n"
"precision highp float;\n"
"layout(location=0) in vec3 aPos;\n"
"uniform mat4 uLightMVP;\n"
"void main(void){\n"
"  gl_Position = uLightMVP * vec4(aPos, 1.0);\n"
"}\n";

static const char* kShadowFs =
"#version 310 es\n"
"precision highp float;\n"
"void main(void){}\n";

static const char* kShadowInstVs =
"#version 310 es\n"
"precision highp float;\n"
"layout(location=0) in vec3 aPos;\n"
"layout(location=3) in mat4 aInstanceModel;\n"
"uniform mat4 uLightVP;\n"
"void main(void){\n"
"  gl_Position = uLightVP * aInstanceModel * vec4(aPos, 1.0);\n"
"}\n";

static const char* kMainVs =
"#version 310 es\n"
"precision highp float;\n"
"layout(location=0) in vec3 aPos;\n"
"layout(location=1) in vec3 aNormal;\n"
"uniform mat4 uMVP;\n"
"uniform mat4 uModel;\n"
"uniform mat4 uLightMVP;\n"
"out vec3 vNormal;\n"
"out float vLocalY;\n"
"out vec4 vPosLight;\n"
"void main(void){\n"
"  gl_Position = uMVP * vec4(aPos,1.0);\n"
"  vNormal = mat3(transpose(inverse(uModel))) * aNormal;\n"
"  vLocalY = aPos.y + 0.5;\n"
"  vPosLight = uLightMVP * vec4(aPos,1.0);\n"
"}\n";

static const char* kMainFs =
"#version 310 es\n"
"precision highp float;\n"
"in vec3 vNormal;\n"
"in float vLocalY;\n"
"in vec4 vPosLight;\n"
"uniform vec3 uColor;\n"
"uniform vec3 uLightDir;\n"
"uniform float uAlpha;\n"
"uniform int uUseFade;\n"
"uniform sampler2D uShadowMap;\n"
"layout(location=0) out vec4 outColor;\n"
"float shadowCalc(vec4 posL){\n"
"  vec3 proj = posL.xyz / max(posL.w, 1e-6);\n"
"  proj = proj * 0.5 + 0.5;\n"
"  if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) return 0.0;\n"
"  float currentDepth = proj.z;\n"
"  float bias = 0.020;\n"
"  vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0));\n"
"  float shadow = 0.0;\n"
"  for (int x = -1; x <= 1; ++x){\n"
"    for (int y = -1; y <= 1; ++y){\n"
"      float pcf = texture(uShadowMap, proj.xy + vec2(float(x), float(y)) * texel).r;\n"
"      shadow += (currentDepth - bias > pcf) ? 1.0 : 0.0;\n"
"    }\n"
"  }\n"
"  return shadow / 9.0;\n"
"}\n"
"void main(void){\n"
"  vec3 n = normalize(vNormal);\n"
"  float diff = max(dot(n, normalize(-uLightDir)), 0.0);\n"
"  float shadow = 0.0;\n"
"  float light = 0.35 + (1.0 - shadow) * diff * 0.65;\n"
"  float a = (uUseFade != 0) ? (1.0 - vLocalY*vLocalY*vLocalY) : uAlpha;\n"
"  vec3 litColor = uColor * light;\n"
"  outColor = vec4(litColor, clamp(a, 0.0, 1.0));\n"
"}\n";

static const char* kInstVs =
"#version 310 es\n"
"precision highp float;\n"
"layout(location=0) in vec3 aPos;\n"
"layout(location=1) in vec3 aNormal;\n"
"layout(location=2) in vec2 aTexCoord;\n"
"layout(location=3) in mat4 aInstanceModel;\n"
"layout(location=7) in vec3 aMatColor;\n"
"layout(location=8) in float aUseTex;\n"
"uniform mat4 uVP;\n"
"uniform mat4 uLightVP;\n"
"out vec3 vNormal;\n"
"out vec4 vPosLight;\n"
"out vec2 vTexCoord;\n"
"out vec3 vMatColor;\n"
"flat out int vUseTex;\n"
"void main(void){\n"
"  mat4 model = aInstanceModel;\n"
"  gl_Position = uVP * model * vec4(aPos, 1.0);\n"
"  vNormal = mat3(transpose(inverse(model))) * aNormal;\n"
"  vPosLight = uLightVP * model * vec4(aPos, 1.0);\n"
"  vTexCoord = aTexCoord;\n"
"  vMatColor = aMatColor;\n"
"  vUseTex = (aUseTex > 0.5) ? 1 : 0;\n"
"}\n";

static const char* kInstFs =
"#version 310 es\n"
"precision highp float;\n"
"in vec3 vNormal;\n"
"in vec4 vPosLight;\n"
"in vec2 vTexCoord;\n"
"in vec3 vMatColor;\n"
"flat in int vUseTex;\n"
"uniform vec3 uLightDir;\n"
"uniform sampler2D uShadowMap;\n"
"uniform sampler2D uDiffuseTex;\n"
"uniform int uHasDiffuseTex;\n"
"uniform int uEnableShadow;\n"
"layout(location=0) out vec4 outColor;\n"
"float shadowCalc(vec4 posL){\n"
"  vec3 proj = posL.xyz / max(posL.w, 1e-6);\n"
"  proj = proj * 0.5 + 0.5;\n"
"  if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) return 0.0;\n"
"  float currentDepth = proj.z;\n"
"  float bias = 0.030;\n"
"  vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0));\n"
"  float shadow = 0.0;\n"
"  for (int x = -1; x <= 1; ++x){\n"
"    for (int y = -1; y <= 1; ++y){\n"
"      float pcf = texture(uShadowMap, proj.xy + vec2(float(x), float(y)) * texel).r;\n"
"      shadow += (currentDepth - bias > pcf) ? 1.0 : 0.0;\n"
"    }\n"
"  }\n"
"  return shadow / 9.0;\n"
"}\n"
"void main(void){\n"
"  vec3 n = normalize(vNormal);\n"
"  float diff = max(dot(n, normalize(-uLightDir)), 0.0);\n"
"  float shadow = 0.0;\n"
"  float light = 0.35 + (1.0 - shadow) * diff * 0.65;\n"
"  vec3 texColor = texture(uDiffuseTex, vTexCoord).rgb;\n"
"  int useTex = (uHasDiffuseTex != 0 && vUseTex != 0) ? 1 : 0;\n"
"  vec3 baseColor = (useTex != 0) ? texColor : vMatColor;\n"
"  vec3 litColor = baseColor * light;\n"
"  outColor = vec4(litColor, 1.0);\n"
"}\n";

static const char* kSkyVs =
"#version 310 es\n"
"precision highp float;\n"
"layout(location=0) in vec3 aPos;\n"
"uniform mat4 uVP;\n"
"out vec3 vDir;\n"
"void main(void){\n"
"  vDir = aPos;\n"
"  vec4 p = uVP * vec4(aPos,1.0);\n"
"  gl_Position = p.xyww;\n"
"}\n";

static const char* kSkyFs =
"#version 310 es\n"
"precision highp float;\n"
"in vec3 vDir;\n"
"layout(location=0) out vec4 outColor;\n"
"void main(void){\n"
"  vec3 d = normalize(vDir);\n"
"  float t = d.y;\n"
"  vec3 zenith = vec3(0.04,0.10,0.26);\n"
"  vec3 horizon = vec3(0.08,0.17,0.34);\n"
"  vec3 ground = vec3(0.03,0.07,0.16);\n"
"  vec3 c = (t >= 0.0) ? mix(horizon, zenith, pow(t, 0.5)) : mix(horizon, ground, pow(-t,0.3));\n"
"  outColor = vec4(c,1.0);\n"
"}\n";

static const float kCubeVerts[] = {
    -0.5f,-0.5f,-0.5f,  0.0f, 0.0f,-1.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f,-0.5f,-0.5f,  0.0f, 0.0f,-1.0f, 1.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f, 0.5f,-0.5f,  0.0f, 0.0f,-1.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f, 0.5f,-0.5f,  0.0f, 0.0f,-1.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f, 0.5f,-0.5f,  0.0f, 0.0f,-1.0f, 0.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f,-0.5f,-0.5f,  0.0f, 0.0f,-1.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,

    -0.5f,-0.5f, 0.5f,  0.0f, 0.0f, 1.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f,-0.5f, 0.5f,  0.0f, 0.0f, 1.0f, 1.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f, 0.5f, 0.5f,  0.0f, 0.0f, 1.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f, 0.5f, 0.5f,  0.0f, 0.0f, 1.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f, 0.5f, 0.5f,  0.0f, 0.0f, 1.0f, 0.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f,-0.5f, 0.5f,  0.0f, 0.0f, 1.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,

    -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f, 0.5f,-0.5f, -1.0f, 0.0f, 0.0f, 1.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f,-0.5f,-0.5f, -1.0f, 0.0f, 0.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f,-0.5f,-0.5f, -1.0f, 0.0f, 0.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f,-0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,

     0.5f, 0.5f, 0.5f,  1.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f, 0.5f,-0.5f,  1.0f, 0.0f, 0.0f, 1.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f,-0.5f,-0.5f,  1.0f, 0.0f, 0.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f,-0.5f,-0.5f,  1.0f, 0.0f, 0.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f,-0.5f, 0.5f,  1.0f, 0.0f, 0.0f, 0.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f, 0.5f, 0.5f,  1.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,

    -0.5f,-0.5f,-0.5f,  0.0f,-1.0f, 0.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f,-0.5f,-0.5f,  0.0f,-1.0f, 0.0f, 1.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f,-0.5f, 0.5f,  0.0f,-1.0f, 0.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f,-0.5f, 0.5f,  0.0f,-1.0f, 0.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f,-0.5f, 0.5f,  0.0f,-1.0f, 0.0f, 0.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f,-0.5f,-0.5f,  0.0f,-1.0f, 0.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,

    -0.5f, 0.5f,-0.5f,  0.0f, 1.0f, 0.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f, 0.5f,-0.5f,  0.0f, 1.0f, 0.0f, 1.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f, 0.5f, 0.5f,  0.0f, 1.0f, 0.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
     0.5f, 0.5f, 0.5f,  0.0f, 1.0f, 0.0f, 1.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f, 0.5f, 0.5f,  0.0f, 1.0f, 0.0f, 0.0f,1.0f, 0.8f,0.8f,0.8f, 0.0f,
    -0.5f, 0.5f,-0.5f,  0.0f, 1.0f, 0.0f, 0.0f,0.0f, 0.8f,0.8f,0.8f, 0.0f
};

static const float kSkyboxVerts[] = {
    -1, 1,-1,  -1,-1,-1,   1,-1,-1,   1,-1,-1,   1, 1,-1,  -1, 1,-1,
    -1,-1, 1,  -1,-1,-1,  -1, 1,-1,  -1, 1,-1,  -1, 1, 1,  -1,-1, 1,
     1,-1,-1,   1,-1, 1,   1, 1, 1,   1, 1, 1,   1, 1,-1,   1,-1,-1,
    -1,-1, 1,  -1, 1, 1,   1, 1, 1,   1, 1, 1,   1,-1, 1,  -1,-1, 1,
    -1, 1,-1,   1, 1,-1,   1, 1, 1,   1, 1, 1,  -1, 1, 1,  -1, 1,-1,
    -1,-1,-1,  -1,-1, 1,   1,-1,-1,   1,-1,-1,  -1,-1, 1,   1,-1, 1
};

static float vec3_dot(const Vec3* a, const Vec3* b) { return (a->x*b->x) + (a->y*b->y) + (a->z*b->z); }
static Vec3 vec3_sub(const Vec3* a, const Vec3* b) { Vec3 r = {a->x-b->x, a->y-b->y, a->z-b->z}; return r; }
static Vec3 vec3_cross(const Vec3* a, const Vec3* b) {
    Vec3 r = {a->y*b->z - a->z*b->y, a->z*b->x - a->x*b->z, a->x*b->y - a->y*b->x};
    return r;
}
static Vec3 vec3_norm(Vec3 v) {
    const float l = sqrtf((v.x*v.x) + (v.y*v.y) + (v.z*v.z));
    if (l > 1e-6f) { v.x /= l; v.y /= l; v.z /= l; }
    return v;
}

static Mat4 mat4_identity(void) {
    Mat4 r = {{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}};
    return r;
}

static Mat4 mat4_mul(const Mat4* a, const Mat4* b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c) {
        for (int rr = 0; rr < 4; ++rr) {
            r.m[c*4 + rr] =
                a->m[0*4 + rr] * b->m[c*4 + 0] +
                a->m[1*4 + rr] * b->m[c*4 + 1] +
                a->m[2*4 + rr] * b->m[c*4 + 2] +
                a->m[3*4 + rr] * b->m[c*4 + 3];
        }
    }
    return r;
}

static Mat4 mat4_translate(float x, float y, float z) {
    Mat4 r = mat4_identity();
    r.m[12] = x; r.m[13] = y; r.m[14] = z;
    return r;
}

static Mat4 mat4_scale(float x, float y, float z) {
    Mat4 r = {{x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1}};
    return r;
}

static Mat4 mat4_rot_x(float a) {
    const float c = cosf(a);
    const float s = sinf(a);
    Mat4 r = {{1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1}};
    return r;
}

static Mat4 mat4_rot_y(float a) {
    const float c = cosf(a);
    const float s = sinf(a);
    Mat4 r = {{c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1}};
    return r;
}

static Mat4 mat4_perspective(float fovyRad, float aspect, float zNear, float zFar) {
    const float f = 1.0f / tanf(fovyRad * 0.5f);
    Mat4 r = {{0}};
    r.m[0] = f / aspect;
    r.m[5] = -f; // flip top and bottom
    r.m[10] = (zFar + zNear) / (zNear - zFar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zFar * zNear) / (zNear - zFar);
    return r;
}

static Mat4 mat4_ortho(float l, float r, float b, float t, float n, float f) {
    Mat4 m = mat4_identity();
    m.m[0] = 2.0f / (r - l);
    m.m[5] = 2.0f / (t - b);
    m.m[10] = -2.0f / (f - n);
    m.m[12] = -(r + l) / (r - l);
    m.m[13] = -(t + b) / (t - b);
    m.m[14] = -(f + n) / (f - n);
    return m;
}

static Mat4 mat4_lookat(Vec3 eye, Vec3 center, Vec3 up) {
    const Vec3 f = vec3_norm(vec3_sub(&center, &eye));
    Vec3 s = vec3_cross(&f, &up);
    s = vec3_norm(s);
    const Vec3 u = vec3_cross(&s, &f);

    Mat4 m = mat4_identity();
    m.m[0] = s.x;  m.m[1] = s.y;  m.m[2]  = s.z;
    m.m[4] = u.x;  m.m[5] = u.y;  m.m[6]  = u.z;
    m.m[8] = -f.x; m.m[9] = -f.y; m.m[10] = -f.z;

    Mat4 t = mat4_translate(-eye.x, -eye.y, -eye.z);
    return mat4_mul(&m, &t);
}

static Mat4 mat4_remove_translation(const Mat4* m) {
    Mat4 r = *m;
    r.m[12] = 0.0f;
    r.m[13] = 0.0f;
    r.m[14] = 0.0f;
    return r;
}

static GLuint compile_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == GL_FALSE) {
        char logbuf[1024] = {0};
        glGetShaderInfoLog(shader, (GLsizei)sizeof(logbuf), NULL, logbuf);
        M_PRINT(M_ZONE_ERROR, "[PORTED] shader compile fail type=0x%x, %s", type, logbuf);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint link_program(GLuint vs, GLuint fs) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    glDeleteShader(vs);
    glDeleteShader(fs);

    if (ok == GL_FALSE) {
        char logbuf[1024] = {0};
        glGetProgramInfoLog(prog, (GLsizei)sizeof(logbuf), NULL, logbuf);
        M_PRINT(M_ZONE_ERROR, "[PORTED] program link fail %s", logbuf);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

static int32_t create_mesh_from_interleaved(PortedMesh* mesh, const float* data, size_t bytes, uint32_t strideFloats) {
    if (mesh == NULL || data == NULL || strideFloats < 6u) {
        return -1;
    }

    glGenVertexArrays(1, &mesh->vao);
    glGenBuffers(1, &mesh->vbo);
    glBindVertexArray(mesh->vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)bytes, data, GL_STATIC_DRAW);

    const GLsizei stride = (GLsizei)(strideFloats * sizeof(float));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));

    if (strideFloats >= 12u) {
        mesh->hasTexAttrib = true;
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));

        glEnableVertexAttribArray(7);
        glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, stride, (void*)(8 * sizeof(float)));

        glEnableVertexAttribArray(8);
        glVertexAttribPointer(8, 1, GL_FLOAT, GL_FALSE, stride, (void*)(11 * sizeof(float)));
    } else {
        mesh->hasTexAttrib = false;
        glDisableVertexAttribArray(2);
        glVertexAttrib2f(2, 0.5f, 0.5f);
        glDisableVertexAttribArray(7);
        glVertexAttrib3f(7, 0.8f, 0.8f, 0.8f);
        glDisableVertexAttribArray(8);
        glVertexAttrib1f(8, 0.0f);
    }

    mesh->vertexCount = (GLsizei)(bytes / (strideFloats * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    return 0;
}

static int32_t create_simple_mesh(PortedMesh* mesh, const float* data, size_t bytes, bool withNormal) {
    glGenVertexArrays(1, &mesh->vao);
    glGenBuffers(1, &mesh->vbo);
    glBindVertexArray(mesh->vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)bytes, data, GL_STATIC_DRAW);

    if (withNormal) {
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12 * (GLsizei)sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12 * (GLsizei)sizeof(float), (void*)(3 * sizeof(float)));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 12 * (GLsizei)sizeof(float), (void*)(6 * sizeof(float)));

        glEnableVertexAttribArray(7);
        glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, 12 * (GLsizei)sizeof(float), (void*)(8 * sizeof(float)));

        glEnableVertexAttribArray(8);
        glVertexAttribPointer(8, 1, GL_FLOAT, GL_FALSE, 12 * (GLsizei)sizeof(float), (void*)(11 * sizeof(float)));

        mesh->hasTexAttrib = true;
        mesh->vertexCount = (GLsizei)(bytes / (12 * sizeof(float)));
    } else {
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * (GLsizei)sizeof(float), (void*)0);
        mesh->vertexCount = (GLsizei)(bytes / (3 * sizeof(float)));
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    return 0;
}

static int32_t load_mesh_bin(const char* path, PortedMesh* mesh) {
    FILE* fp = fopen(path, "rb");
    if (fp == NULL) {
        M_PRINT(M_ZONE_ERROR, "[PORTED] mesh open fail %s", path);
        return -1;
    }

    MeshBinHeader header;
    if (fread(&header, sizeof(header), 1u, fp) != 1u) {
        fclose(fp);
        M_PRINT(M_ZONE_ERROR, "[PORTED] mesh header read fail %s", path);
        return -1;
    }

    if (header.magic != MESH_MAGIC || (header.strideFloats != 6u && header.strideFloats != 12u)) {
        fclose(fp);
        M_PRINT(M_ZONE_ERROR, "[PORTED] mesh format fail %s", path);
        return -1;
    }

    const size_t floatCount = (size_t)header.strideFloats * (size_t)header.vertexCount;
    float* vertices = (float*)malloc(floatCount * sizeof(float));
    if (vertices == NULL) {
        fclose(fp);
        return -1;
    }

    if (fread(vertices, sizeof(float), floatCount, fp) != floatCount) {
        free(vertices);
        fclose(fp);
        M_PRINT(M_ZONE_ERROR, "[PORTED] mesh payload read fail %s", path);
        return -1;
    }

    fclose(fp);
    memset(mesh, 0, sizeof(*mesh));
    int32_t rc = create_mesh_from_interleaved(mesh, vertices, floatCount * sizeof(float), header.strideFloats);
    free(vertices);
    return rc;
}

static GLuint load_texture_bin(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (fp == NULL) {
        M_PRINT(M_ZONE_ERROR, "[PORTED] tex open fail %s", path);
        return 0u;
    }

    TexBinHeader header;
    if (fread(&header, sizeof(header), 1u, fp) != 1u) {
        fclose(fp);
        return 0u;
    }

    if (header.magic != TEX_MAGIC || header.channels < 1u || header.channels > 4u) {
        fclose(fp);
        M_PRINT(M_ZONE_ERROR, "[PORTED] tex format fail %s", path);
        return 0u;
    }

    const size_t pixelCount = (size_t)header.width * (size_t)header.height * (size_t)header.channels;
    uint8_t* pixels = (uint8_t*)malloc(pixelCount);
    if (pixels == NULL) {
        fclose(fp);
        return 0u;
    }

    if (fread(pixels, 1u, pixelCount, fp) != pixelCount) {
        free(pixels);
        fclose(fp);
        return 0u;
    }
    fclose(fp);

    GLenum format = GL_RGB;
    if (header.channels == 1u) format = GL_RED;
    else if (header.channels == 4u) format = GL_RGBA;

    GLuint tex = 0u;
    GLint prevUnpackAlignment = 4;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevUnpackAlignment);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)format, (GLsizei)header.width, (GLsizei)header.height, 0, format, GL_UNSIGNED_BYTE, pixels);
    glPixelStorei(GL_UNPACK_ALIGNMENT, prevUnpackAlignment);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    free(pixels);
    return tex;
}

static int32_t create_shadow_target(void) {
    glGenFramebuffers(1, &g_ported.depthFbo);
    glGenTextures(1, &g_ported.depthTex);

    glBindTexture(GL_TEXTURE_2D, g_ported.depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, PORTED_SHADOW_W, PORTED_SHADOW_H, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindFramebuffer(GL_FRAMEBUFFER, g_ported.depthFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_ported.depthTex, 0);
    {
        const GLenum noneBuf = GL_NONE;
        glDrawBuffers(1, &noneBuf);
    }
    glReadBuffer(GL_NONE);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        M_PRINT(M_ZONE_ERROR, "[PORTED] shadow FBO incomplete");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return -1;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return 0;
}

static uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static void set_instanced_attribs(GLuint vao, GLuint instBuf) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, instBuf);
    for (int i = 0; i < 4; ++i) {
        glEnableVertexAttribArray((GLuint)(3 + i));
        glVertexAttribPointer((GLuint)(3 + i), 4, GL_FLOAT, GL_FALSE, (GLsizei)sizeof(Mat4), (void*)(sizeof(float) * 4 * i));
        glVertexAttribDivisor((GLuint)(3 + i), 1);
    }
    glBindVertexArray(0);
}

static void draw_shadow_cube(const Mat4* matLightVP, const Mat4* matModel) {
    const Mat4 matLightMVP = mat4_mul(matLightVP, matModel);
    glUniformMatrix4fv(glGetUniformLocation(g_ported.shadowProgram, "uLightMVP"), 1, GL_FALSE, matLightMVP.m);
    glBindVertexArray(g_ported.cubeMesh.vao);
    glDrawArrays(GL_TRIANGLES, 0, g_ported.cubeMesh.vertexCount);
}

static void draw_main_model(const Mat4* matViewProj, const Mat4* matLightVP, const Mat4* matModel, const Vec3* pTintColor, int fadeEnabled, GLuint vaoId, GLsizei vertexCount) {
    const Mat4 matMVP = mat4_mul(matViewProj, matModel);
    const Mat4 matLightMVP = mat4_mul(matLightVP, matModel);

    glUniformMatrix4fv(glGetUniformLocation(g_ported.mainProgram, "uMVP"), 1, GL_FALSE, matMVP.m);
    glUniformMatrix4fv(glGetUniformLocation(g_ported.mainProgram, "uModel"), 1, GL_FALSE, matModel->m);
    glUniformMatrix4fv(glGetUniformLocation(g_ported.mainProgram, "uLightMVP"), 1, GL_FALSE, matLightMVP.m);
    glUniform3f(glGetUniformLocation(g_ported.mainProgram, "uColor"), pTintColor->x, pTintColor->y, pTintColor->z);
    glUniform1f(glGetUniformLocation(g_ported.mainProgram, "uAlpha"), 1.0f);
    glUniform1i(glGetUniformLocation(g_ported.mainProgram, "uUseFade"), fadeEnabled);

    glBindVertexArray(vaoId);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
}

static void draw_parking_strip(const Mat4* matViewProj,
                               const Mat4* matLightVP,
                               float cx,
                               float cz,
                               float yaw,
                               float lx,
                               float lz,
                               float angle,
                               float sx,
                               float sz,
                               const Vec3* color) {
    Mat4 model = mat4_identity();
    const Mat4 t0 = mat4_translate(cx, 0.08f, cz);
    const Mat4 r0 = mat4_rot_y(yaw);
    const Mat4 t1 = mat4_translate(lx, 0.0f, lz);
    const Mat4 r1 = mat4_rot_y(angle);
    const Mat4 s = mat4_scale(sx, 0.01f, sz);

    model = mat4_mul(&model, &t0);
    model = mat4_mul(&model, &r0);
    model = mat4_mul(&model, &t1);
    model = mat4_mul(&model, &r1);
    model = mat4_mul(&model, &s);

    draw_main_model(matViewProj, matLightVP, &model, color, 0, g_ported.cubeMesh.vao, g_ported.cubeMesh.vertexCount);
}

static void draw_parking_spot(const Mat4* matViewProj, const Mat4* matLightVP, float cx, float cz, float yaw) {
    const float halfW = 2.0f;
    const float halfL = 3.5f;
    const float r = 0.5f;
    const float lw = 0.15f;
    const int segCount = 8;
    const float straightL = (halfL - r) * 2.0f;
    const float straightW = (halfW - r) * 2.0f;
    const float arcSegLen = r * ((float)M_PI * 0.5f) / (float)segCount * 1.1f;
    const Vec3 color = {0.42f, 0.44f, 0.45f};

    draw_parking_strip(matViewProj, matLightVP, cx, cz, yaw, -halfW, 0.0f, 0.0f, lw, straightL, &color);
    draw_parking_strip(matViewProj, matLightVP, cx, cz, yaw,  halfW, 0.0f, 0.0f, lw, straightL, &color);
    draw_parking_strip(matViewProj, matLightVP, cx, cz, yaw, 0.0f, -halfL, 0.0f, straightW, lw, &color);
    draw_parking_strip(matViewProj, matLightVP, cx, cz, yaw, 0.0f,  halfL, 0.0f, straightW, lw, &color);

    const float ox0 = -(halfW - r), oz0 = -(halfL - r), a0 = (float)M_PI;
    const float ox1 =  (halfW - r), oz1 = -(halfL - r), a1 = (float)M_PI * 1.5f;
    const float ox2 =  (halfW - r), oz2 =  (halfL - r), a2 = 0.0f;
    const float ox3 = -(halfW - r), oz3 =  (halfL - r), a3 = (float)M_PI * 0.5f;

    for (int j = 0; j < segCount; ++j) {
        const float t = ((float)j + 0.5f) / (float)segCount;
        const float da = t * ((float)M_PI * 0.5f);

        const float alpha0 = a0 + da;
        draw_parking_strip(matViewProj, matLightVP, cx, cz, yaw,
                          ox0 + r * cosf(alpha0), oz0 + r * sinf(alpha0), alpha0, lw, arcSegLen, &color);

        const float alpha1 = a1 + da;
        draw_parking_strip(matViewProj, matLightVP, cx, cz, yaw,
                          ox1 + r * cosf(alpha1), oz1 + r * sinf(alpha1), alpha1, lw, arcSegLen, &color);

        const float alpha2 = a2 + da;
        draw_parking_strip(matViewProj, matLightVP, cx, cz, yaw,
                          ox2 + r * cosf(alpha2), oz2 + r * sinf(alpha2), alpha2, lw, arcSegLen, &color);

        const float alpha3 = a3 + da;
        draw_parking_strip(matViewProj, matLightVP, cx, cz, yaw,
                          ox3 + r * cosf(alpha3), oz3 + r * sinf(alpha3), alpha3, lw, arcSegLen, &color);
    }
}

static void draw_ground_slab(const Mat4* matViewProj,
                             const Mat4* matLightVP,
                             const Vec3* carPos,
                             bool shadowOnly) {
    Mat4 tr = mat4_translate(carPos->x, 0.0f, carPos->z);
    Mat4 sc = mat4_scale(PORTED_GROUND_RADIUS * 2.0f, 0.1f, PORTED_GROUND_RADIUS * 2.0f);
    Mat4 m = mat4_mul(&tr, &sc);

    if (shadowOnly) {
        draw_shadow_cube(matLightVP, &m);
    } else {
        const Vec3 c = {0.07f, 0.09f, 0.13f};
        draw_main_model(matViewProj, matLightVP, &m, &c, 0, g_ported.cubeMesh.vao, g_ported.cubeMesh.vertexCount);
    }
}

int32_t mpa_ported_demo_start(void) {
    if (g_ported.initialized) {
        return 0;
    }

    memset(&g_ported, 0, sizeof(g_ported));

    g_ported.mainProgram = link_program(compile_shader(GL_VERTEX_SHADER, kMainVs), compile_shader(GL_FRAGMENT_SHADER, kMainFs));
    g_ported.shadowProgram = link_program(compile_shader(GL_VERTEX_SHADER, kShadowVs), compile_shader(GL_FRAGMENT_SHADER, kShadowFs));
    g_ported.shadowInstProgram = link_program(compile_shader(GL_VERTEX_SHADER, kShadowInstVs), compile_shader(GL_FRAGMENT_SHADER, kShadowFs));
    g_ported.instProgram = link_program(compile_shader(GL_VERTEX_SHADER, kInstVs), compile_shader(GL_FRAGMENT_SHADER, kInstFs));
    g_ported.skyProgram = link_program(compile_shader(GL_VERTEX_SHADER, kSkyVs), compile_shader(GL_FRAGMENT_SHADER, kSkyFs));

    if ((g_ported.mainProgram == 0u) || (g_ported.shadowProgram == 0u) || (g_ported.shadowInstProgram == 0u) || (g_ported.instProgram == 0u) || (g_ported.skyProgram == 0u)) {
        M_PRINT(M_ZONE_ERROR, "[PORTED] shader program create failed");
        mpa_ported_demo_stop();
        return -1;
    }

    create_simple_mesh(&g_ported.cubeMesh, kCubeVerts, sizeof(kCubeVerts), true);
    create_simple_mesh(&g_ported.skyMesh, kSkyboxVerts, sizeof(kSkyboxVerts), false);

    if (load_mesh_bin("ioniq_5.meshbin", &g_ported.carMesh) != 0) {
        g_ported.carMesh = g_ported.cubeMesh;
        g_ported.usingFallbackCar = true;
        M_PRINT(M_ZONE_LOG, "[PORTED] fallback car mesh");
    }

    if (load_mesh_bin("12353_Automobile_V1_L2.meshbin", &g_ported.parkedMesh) != 0) {
        g_ported.parkedMesh = g_ported.cubeMesh;
        g_ported.usingFallbackParked = true;
        M_PRINT(M_ZONE_LOG, "[PORTED] fallback parked mesh");
    } else {
        g_ported.parkedMesh.diffuseTex = load_texture_bin("Body_diff.texbin");
        if (g_ported.parkedMesh.diffuseTex == 0u) {
            g_ported.usingFallbackParked = true;
            M_PRINT(M_ZONE_LOG, "[PORTED] fallback parked texture color-only");
        }
    }

    glGenBuffers(1, &g_ported.instModelBuf);
    if (g_ported.instModelBuf == 0u) {
        mpa_ported_demo_stop();
        return -1;
    }
    set_instanced_attribs(g_ported.parkedMesh.vao, g_ported.instModelBuf);

    if (create_shadow_target() != 0) {
        mpa_ported_demo_stop();
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    g_ported.carPos[0] = 0.0f;
    g_ported.carPos[1] = 0.45f;
    g_ported.carPos[2] = 0.0f;
    g_ported.smoothCamPos[0] = 0.0f;
    g_ported.smoothCamPos[1] = 5.0f;
    g_ported.smoothCamPos[2] = -10.0f;

    g_ported.initialized = true;
    M_PRINT(M_ZONE_LOG, "[PORTED] started");
    return 0;
}

void mpa_ported_demo_render(int32_t width, int32_t height) {
    if (!g_ported.initialized || width < 0 || height < 0) {
        return;
    }

    GLint savedFramebuffer = 0;
    GLint savedViewport[4] = {0};
    GLint savedProgram = 0;
    GLint savedArrayBuffer = 0;
    GLint savedVao = 0;
    GLint savedActiveTexture = 0;
    GLint savedTex0 = 0;
    GLint savedTex1 = 0;
    GLint savedDepthFunc = GL_LESS;
    GLint savedBlendSrcRgb = GL_ONE;
    GLint savedBlendDstRgb = GL_ZERO;
    GLint savedBlendSrcAlpha = GL_ONE;
    GLint savedBlendDstAlpha = GL_ZERO;
    GLint savedFrontFace = GL_CCW;
    GLboolean savedDepthMask = GL_TRUE;
    GLboolean savedColorMask[4] = {GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE};
    GLboolean savedDepthTest = glIsEnabled(GL_DEPTH_TEST);
    GLboolean savedBlend = glIsEnabled(GL_BLEND);
    GLboolean savedCull = glIsEnabled(GL_CULL_FACE);

    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &savedFramebuffer);
    glGetIntegerv(GL_VIEWPORT, savedViewport);
    glGetIntegerv(GL_CURRENT_PROGRAM, &savedProgram);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &savedArrayBuffer);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &savedVao);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &savedActiveTexture);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &savedDepthMask);
    glGetBooleanv(GL_COLOR_WRITEMASK, savedColorMask);
    glGetIntegerv(GL_DEPTH_FUNC, &savedDepthFunc);
    glGetIntegerv(GL_BLEND_SRC_RGB, &savedBlendSrcRgb);
    glGetIntegerv(GL_BLEND_DST_RGB, &savedBlendDstRgb);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &savedBlendSrcAlpha);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &savedBlendDstAlpha);
    glGetIntegerv(GL_FRONT_FACE, &savedFrontFace);

    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTex0);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glActiveTexture(GL_TEXTURE1);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTex1);
    glActiveTexture(savedActiveTexture);

    const float dt = 1.0f / 30.0f;
    g_ported.timeSec += dt;

    {
        const float idleSpeed = 3.0f;
        if (g_ported.carSpeed < idleSpeed) {
            g_ported.carSpeed = fminf(idleSpeed, g_ported.carSpeed + idleSpeed * dt * 6.0f);
        }

        const float drag = 8.0f;
        const float dragForce = drag * dt;
        if (fabsf(g_ported.carSpeed) <= dragForce) {
            g_ported.carSpeed = 0.0f;
        } else {
            g_ported.carSpeed -= dragForce * (g_ported.carSpeed > 0.0f ? 1.0f : -1.0f);
        }

        const float targetSteering = 0.0f;
        const float steeringReturn = 12.0f;
        if (fabsf(targetSteering - g_ported.steeringInput) > 0.01f) {
            const float step = steeringReturn * dt;
            if (targetSteering > g_ported.steeringInput) {
                g_ported.steeringInput = fminf(targetSteering, g_ported.steeringInput + step);
            } else {
                g_ported.steeringInput = fmaxf(targetSteering, g_ported.steeringInput - step);
            }
        } else {
            g_ported.steeringInput = targetSteering;
        }

        const Vec3 forward = {sinf(g_ported.carRotation), 0.0f, cosf(g_ported.carRotation)};
        g_ported.carPos[0] += forward.x * g_ported.carSpeed * dt;
        g_ported.carPos[2] += forward.z * g_ported.carSpeed * dt;
    }

    const Vec3 carPos = {g_ported.carPos[0], g_ported.carPos[1], g_ported.carPos[2]};
    const Vec3 forward = {sinf(g_ported.carRotation), 0.0f, cosf(g_ported.carRotation)};
    const Vec3 carCenter = {carPos.x, carPos.y + 0.5f, carPos.z};
    const Vec3 targetCam = {carCenter.x - forward.x * 13.0f, carCenter.y + 7.5f, carCenter.z - forward.z * 13.0f};

    {
        const float camAlpha = 1.0f - powf(0.001f, dt);
        g_ported.smoothCamPos[0] += (targetCam.x - g_ported.smoothCamPos[0]) * camAlpha;
        g_ported.smoothCamPos[1] += (targetCam.y - g_ported.smoothCamPos[1]) * camAlpha;
        g_ported.smoothCamPos[2] += (targetCam.z - g_ported.smoothCamPos[2]) * camAlpha;
    }

    const Vec3 camPos = {g_ported.smoothCamPos[0], g_ported.smoothCamPos[1], g_ported.smoothCamPos[2]};
    Vec3 lightDir = {0.4f, -1.0f, 0.3f};
    lightDir = vec3_norm(lightDir);
    const Vec3 lightPos = {carPos.x - lightDir.x * 60.0f, carPos.y + 60.0f - lightDir.y * 60.0f, carPos.z - lightDir.z * 60.0f};

    const int32_t viewW = (width > 0) ? width : savedViewport[2];
    const int32_t viewH = (height > 0) ? height : savedViewport[3];
    const Mat4 view = mat4_lookat(camPos, carCenter, (Vec3){0.0f, 1.0f, 0.0f});
    const Mat4 proj = mat4_perspective((float)(60.0 * M_PI / 180.0), (float)viewW / (float)viewH, 0.1f, 1000.0f);
    const Mat4 vp = mat4_mul(&proj, &view);

    const Mat4 lightView = mat4_lookat(lightPos, carPos, (Vec3){0.0f, 1.0f, 0.0f});
    const Mat4 lightProj = mat4_ortho(-80.0f, 80.0f, -80.0f, 80.0f, 1.0f, 200.0f);
    const Mat4 lightVP = mat4_mul(&lightProj, &lightView);

    Mat4 instModels[PORTED_MAX_INSTANCES];
    int instCount = 0;
    {
        const float spawnRadius = 40.0f;
        const float slotSpacing = 8.0f;
        const int minI = (int)floorf((carPos.z - spawnRadius) / slotSpacing) - 1;
        const int maxI = (int)ceilf((carPos.z + spawnRadius) / slotSpacing) + 1;

        for (int i = minI; i <= maxI; ++i) {
            const float z = (float)i * slotSpacing;
            for (int side = 0; side < 2; ++side) {
                const float x = (side == 0) ? -10.0f : 10.0f;
                const float yaw = (side == 0) ? ((float)M_PI * 0.5f) : (-(float)M_PI * 0.5f);

                uint32_t h = (uint32_t)(i * 2 + side) * 2654435761u;
                if ((h % 3u) == 0u) {
                    continue;
                }

                const float dx = x - carPos.x;
                const float dz = z - carPos.z;
                if ((dx * dx + dz * dz) > (spawnRadius * spawnRadius)) {
                    continue;
                }

                if (instCount >= PORTED_MAX_INSTANCES) {
                    continue;
                }

                Mat4 m = mat4_identity();
                const Mat4 t = mat4_translate(x, 0.05f, z);
                const Mat4 rY = mat4_rot_y(yaw + ((float)M_PI * 0.5f));
                const Mat4 rX = mat4_rot_x(-(float)M_PI * 0.5f);
                const Mat4 s = mat4_scale(0.014f, 0.014f, 0.014f);
                const Mat4 t2 = mat4_translate(-7.7655f, 0.0f, 0.6587f);

                m = mat4_mul(&m, &t);
                m = mat4_mul(&m, &rY);
                m = mat4_mul(&m, &rX);
                m = mat4_mul(&m, &s);
                m = mat4_mul(&m, &t2);

                instModels[instCount++] = m;
            }
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, g_ported.instModelBuf);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(Mat4) * (size_t)instCount), instModels, GL_DYNAMIC_DRAW);

    Pillar pillars[PORTED_MAX_PILLARS];
    int pillarCount = 0;
    {
        const float pillarSpawnRadius = 40.0f;
        const float slotSpacing = 8.0f;
        const int minI = (int)floorf((carPos.z - pillarSpawnRadius) / slotSpacing) - 1;
        const int maxI = (int)ceilf((carPos.z + pillarSpawnRadius) / slotSpacing) + 1;

        for (int i = minI; i <= maxI; ++i) {
            const float baseZ = (float)i * slotSpacing + 4.0f;
            const float dz = baseZ - carPos.z;
            if ((dz * dz) > (pillarSpawnRadius * pillarSpawnRadius)) {
                continue;
            }

            for (int side = 0; side < 2; ++side) {
                uint32_t h = hash_u32((uint32_t)(i ^ (side * 0x9e3779b9)));
                if ((h & 3u) == 0u) {
                    continue;
                }
                if (pillarCount >= PORTED_MAX_PILLARS) {
                    continue;
                }
                pillars[pillarCount].x = (side == 0) ? -7.0f : 7.0f;
                pillars[pillarCount].z = baseZ;
                ++pillarCount;
            }
        }
    }

    if (PORTED_ENABLE_SHADOWS) {
        glViewport(0, 0, PORTED_SHADOW_W, PORTED_SHADOW_H);
        glBindFramebuffer(GL_FRAMEBUFFER, g_ported.depthFbo);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(2.0f, 4.0f);

        glUseProgram(g_ported.shadowProgram);

        {
            draw_ground_slab(&vp, &lightVP, &carPos, true);
        }

        {
            Mat4 carModel = mat4_identity();
            Mat4 t = mat4_translate(carPos.x, carPos.y, carPos.z);
            Mat4 r = mat4_rot_y(g_ported.carRotation - ((float)M_PI * 0.5f));
            Mat4 s = g_ported.usingFallbackCar ? mat4_scale(1.9f, 0.8f, 4.2f) : mat4_scale(0.05f, 0.05f, 0.05f);
            Mat4 t2 = g_ported.usingFallbackCar ? mat4_identity() : mat4_translate(-39.35f, 0.0f, 0.0f);
            carModel = mat4_mul(&carModel, &t);
            carModel = mat4_mul(&carModel, &r);
            carModel = mat4_mul(&carModel, &s);
            carModel = mat4_mul(&carModel, &t2);

            Mat4 lmvp = mat4_mul(&lightVP, &carModel);
            glUniformMatrix4fv(glGetUniformLocation(g_ported.shadowProgram, "uLightMVP"), 1, GL_FALSE, lmvp.m);
            glEnable(GL_CULL_FACE);
            glFrontFace(GL_CCW);
            glCullFace(GL_BACK);
            glBindVertexArray(g_ported.carMesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, g_ported.carMesh.vertexCount);
            glFrontFace(GL_CW);
            glDisable(GL_CULL_FACE);
        }

        {
            for (int i = 0; i < pillarCount; ++i) {
                Mat4 tr = mat4_translate(pillars[i].x, 3.0f, pillars[i].z);
                Mat4 sc = mat4_scale(1.0f, 6.0f, 1.0f);
                Mat4 m = mat4_mul(&tr, &sc);
                draw_shadow_cube(&lightVP, &m);
            }
        }

        if (instCount > 0) {
            glUseProgram(g_ported.shadowInstProgram);
            glUniformMatrix4fv(glGetUniformLocation(g_ported.shadowInstProgram, "uLightVP"), 1, GL_FALSE, lightVP.m);
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
            glBindVertexArray(g_ported.parkedMesh.vao);
            glDrawArraysInstanced(GL_TRIANGLES, 0, g_ported.parkedMesh.vertexCount, instCount);
            glDisable(GL_CULL_FACE);
        }

        glDisable(GL_POLYGON_OFFSET_FILL);
        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)savedFramebuffer);

    }

    const int32_t renderW = (width > 0) ? width : savedViewport[2];
    const int32_t renderH = (height > 0) ? height : savedViewport[3];
    glViewport(savedViewport[0], savedViewport[1], renderW, renderH);
    glClearColor(0.02f, 0.03f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LEQUAL);
    glUseProgram(g_ported.skyProgram);
    {
        const Mat4 skyView = mat4_remove_translation(&view);
        const Mat4 skyVP = mat4_mul(&proj, &skyView);
        glUniformMatrix4fv(glGetUniformLocation(g_ported.skyProgram, "uVP"), 1, GL_FALSE, skyVP.m);
        glBindVertexArray(g_ported.skyMesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, g_ported.skyMesh.vertexCount);
    }
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
    glDisable(GL_BLEND);
    glFrontFace(GL_CW);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_ported.depthTex);

    glUseProgram(g_ported.mainProgram);
    glUniform3f(glGetUniformLocation(g_ported.mainProgram, "uLightDir"), lightDir.x, lightDir.y, lightDir.z);
    glUniform1i(glGetUniformLocation(g_ported.mainProgram, "uShadowMap"), 0);

    {
        draw_ground_slab(&vp, &lightVP, &carPos, false);
    }

    {
        const float spawnRadius = 40.0f;
        const float slotSpacing = 8.0f;
        const int minI = (int)floorf((carPos.z - spawnRadius) / slotSpacing) - 1;
        const int maxI = (int)ceilf((carPos.z + spawnRadius) / slotSpacing) + 1;
        for (int i = minI; i <= maxI; ++i) {
            const float z = (float)i * slotSpacing;
            const float dz = z - carPos.z;
            const float dxL = -10.0f - carPos.x;
            const float dxR = 10.0f - carPos.x;
            if ((dxL * dxL + dz * dz) <= (spawnRadius * spawnRadius)) {
                draw_parking_spot(&vp, &lightVP, -10.0f, z, (float)M_PI * 0.5f);
            }
            if ((dxR * dxR + dz * dz) <= (spawnRadius * spawnRadius)) {
                draw_parking_spot(&vp, &lightVP, 10.0f, z, -(float)M_PI * 0.5f);
            }
        }
    }

    if (instCount > 0) {
        glUseProgram(g_ported.instProgram);
        glUniformMatrix4fv(glGetUniformLocation(g_ported.instProgram, "uVP"), 1, GL_FALSE, vp.m);
        glUniformMatrix4fv(glGetUniformLocation(g_ported.instProgram, "uLightVP"), 1, GL_FALSE, lightVP.m);
        glUniform3f(glGetUniformLocation(g_ported.instProgram, "uLightDir"), lightDir.x, lightDir.y, lightDir.z);
        /* Bind diffuse to unit 0 (shadow map slot) as well so no depth texture
           sits in any sampler2D slot - depth+sampler2D can misbehave on Adreno. */
        if (g_ported.parkedMesh.diffuseTex != 0u) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, g_ported.parkedMesh.diffuseTex);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, g_ported.parkedMesh.diffuseTex);
            glActiveTexture(GL_TEXTURE0);
        }
        glUniform1i(glGetUniformLocation(g_ported.instProgram, "uShadowMap"), 0);
        glUniform1i(glGetUniformLocation(g_ported.instProgram, "uDiffuseTex"), 1);
        glUniform1i(glGetUniformLocation(g_ported.instProgram, "uHasDiffuseTex"), (g_ported.parkedMesh.diffuseTex != 0u) ? 1 : 0);
        glUniform1i(glGetUniformLocation(g_ported.instProgram, "uEnableShadow"), 0);


        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glBindVertexArray(g_ported.parkedMesh.vao);
        glDrawArraysInstanced(GL_TRIANGLES, 0, g_ported.parkedMesh.vertexCount, instCount);
        glDisable(GL_CULL_FACE);
        /* Restore shadow depth texture to unit 0 for subsequent main-program draws. */
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_ported.depthTex);
    }

    glUseProgram(g_ported.mainProgram);
    {
        Vec3 c = {0.80f, 0.80f, 0.80f};
        Mat4 carModel = mat4_identity();
        Mat4 t = mat4_translate(carPos.x, carPos.y, carPos.z);
        Mat4 r = mat4_rot_y(g_ported.carRotation - ((float)M_PI * 0.5f));
        Mat4 s = g_ported.usingFallbackCar ? mat4_scale(1.9f, 0.8f, 4.2f) : mat4_scale(0.05f, 0.05f, 0.05f);
        Mat4 t2 = g_ported.usingFallbackCar ? mat4_identity() : mat4_translate(-39.35f, 0.0f, 0.0f);
        carModel = mat4_mul(&carModel, &t);
        carModel = mat4_mul(&carModel, &r);
        carModel = mat4_mul(&carModel, &s);
        carModel = mat4_mul(&carModel, &t2);
        glEnable(GL_CULL_FACE);
        // glFrontFace(GL_CW);
        glCullFace(GL_BACK);
        /* depth pre-pass: prime the depth buffer with frontmost surfaces */
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDepthFunc(GL_LESS);
        draw_main_model(&vp, &lightVP, &carModel, &c, 0, g_ported.carMesh.vao, g_ported.carMesh.vertexCount);
        /* color pass: only the already-primed closest fragments pass */
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthFunc(GL_LEQUAL);
        draw_main_model(&vp, &lightVP, &carModel, &c, 0, g_ported.carMesh.vao, g_ported.carMesh.vertexCount);
        glDepthFunc(GL_LESS);
        glDisable(GL_CULL_FACE);

    }

    for (int i = 0; i < pillarCount; ++i) {
        Vec3 c = {0.70f, 0.70f, 0.70f};
        Mat4 tr = mat4_translate(pillars[i].x, 3.0f, pillars[i].z);
        Mat4 sc = mat4_scale(1.0f, 6.0f, 1.0f);
        Mat4 m = mat4_mul(&tr, &sc);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        glDepthMask(GL_TRUE);
        draw_main_model(&vp, &lightVP, &m, &c, 1, g_ported.cubeMesh.vao, g_ported.cubeMesh.vertexCount);

        glCullFace(GL_BACK);
        glDepthMask(GL_FALSE);
        draw_main_model(&vp, &lightVP, &m, &c, 1, g_ported.cubeMesh.vao, g_ported.cubeMesh.vertexCount);

        glDisable(GL_CULL_FACE);
        glDepthMask(GL_TRUE);
    }

    glDisable(GL_BLEND);

    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)savedFramebuffer);
    glViewport(savedViewport[0], savedViewport[1], savedViewport[2], savedViewport[3]);

    if (savedDepthTest) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    if (savedBlend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    if (savedCull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    glFrontFace((GLenum)savedFrontFace);
    glBlendFuncSeparate((GLenum)savedBlendSrcRgb, (GLenum)savedBlendDstRgb,
                        (GLenum)savedBlendSrcAlpha, (GLenum)savedBlendDstAlpha);
    glDepthMask(savedDepthMask);
    glColorMask(savedColorMask[0], savedColorMask[1], savedColorMask[2], savedColorMask[3]);
    glDepthFunc((GLenum)savedDepthFunc);

    glUseProgram((GLuint)savedProgram);
    glBindVertexArray((GLuint)savedVao);
    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)savedArrayBuffer);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, (GLuint)savedTex0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, (GLuint)savedTex1);
    glActiveTexture(savedActiveTexture);
}


void mpa_ported_demo_stop(void) {
    if (!g_ported.initialized) {
        return;
    }

    if (g_ported.depthFbo != 0u) glDeleteFramebuffers(1, &g_ported.depthFbo);
    if (g_ported.depthTex != 0u) glDeleteTextures(1, &g_ported.depthTex);

    if (g_ported.instModelBuf != 0u) glDeleteBuffers(1, &g_ported.instModelBuf);

    if (g_ported.cubeMesh.vbo != 0u) glDeleteBuffers(1, &g_ported.cubeMesh.vbo);
    if (g_ported.cubeMesh.vao != 0u) glDeleteVertexArrays(1, &g_ported.cubeMesh.vao);
    if (g_ported.skyMesh.vbo != 0u) glDeleteBuffers(1, &g_ported.skyMesh.vbo);
    if (g_ported.skyMesh.vao != 0u) glDeleteVertexArrays(1, &g_ported.skyMesh.vao);

    if (!g_ported.usingFallbackCar) {
        if (g_ported.carMesh.vbo != 0u) glDeleteBuffers(1, &g_ported.carMesh.vbo);
        if (g_ported.carMesh.vao != 0u) glDeleteVertexArrays(1, &g_ported.carMesh.vao);
    }

    if (!g_ported.usingFallbackParked) {
        if (g_ported.parkedMesh.vbo != 0u) glDeleteBuffers(1, &g_ported.parkedMesh.vbo);
        if (g_ported.parkedMesh.vao != 0u) glDeleteVertexArrays(1, &g_ported.parkedMesh.vao);
    }
    if (g_ported.parkedMesh.diffuseTex != 0u) glDeleteTextures(1, &g_ported.parkedMesh.diffuseTex);

    if (g_ported.mainProgram != 0u) glDeleteProgram(g_ported.mainProgram);
    if (g_ported.shadowProgram != 0u) glDeleteProgram(g_ported.shadowProgram);
    if (g_ported.shadowInstProgram != 0u) glDeleteProgram(g_ported.shadowInstProgram);
    if (g_ported.instProgram != 0u) glDeleteProgram(g_ported.instProgram);
    if (g_ported.skyProgram != 0u) glDeleteProgram(g_ported.skyProgram);

    memset(&g_ported, 0, sizeof(g_ported));
    M_PRINT(M_ZONE_LOG, "[PORTED] stopped");
}

void mpa_ported_demo_run(int32_t width, int32_t height) {
    if (!g_ported.initialized) {
        if (mpa_ported_demo_start() != 0) {
            return;
        }
    }
    mpa_ported_demo_render(width, height);
}
