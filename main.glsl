#version 300 es

precision highp float;
precision highp sampler2D;

in vec2 uv;
out vec4 out_color;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec4 u_mouse;
uniform sampler2D u_textures[16];

const float epsilon = 0.001;
const float culling_distance = 10.;
const int max_steps = 64;

#define PI 3.14159
#define saturate(x) clamp(x, 0.0, 1.0)

vec3 rotate(vec3 p) {
    float angle = u_time;
    vec4 q = vec4(0.0, sin(angle / 2.), 0., cos(angle/2.));
    vec3 qvec = q.xyz;
    vec3 uv = cross(qvec, p);
    vec3 uuv = cross(qvec, uv);
    uv *= (2.0 * q.w);
    uuv *= 2.0;
    return p + uv + uuv;
}

float smin(float a, float b, float k) {
    float h = saturate(0.5 + 0.5*(a-b)/k);
    return mix(a, b, h) - k*h*(1.0-h);
}

float smax(float a, float b, float k) {
    return smin(a, b, -k);
}

float sabs(float a, float k) {
    return sqrt(a*a + k);
}

vec3 sabs(vec3 p, float k) {
    return vec3(sqrt(pow(p, vec3(2.))+k));
}

struct SdfResult {
    float dist;
    float mat;
};

float nelength(vec3 p) {
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

float nedot(vec3 a, vec3 b) {
    float similarity=dot(normalize(a), normalize(b));
    return nelength(a)*nelength(b)*similarity;
}

SdfResult sdfPlane(vec3 p, vec4 n, float mat)
{
    return SdfResult(dot(p, n.xyz) + n.w, mat);
}

SdfResult sdfBox(vec3 pos, vec3 size, vec3 offset, float mat) {
    pos = rotate(pos-offset);
    // pos = pos-offset;
    vec3 dist = abs(pos) - size;
    return SdfResult(length(max(dist, 0.0)) + min(max(dist.x, max(dist.y, dist.z)), 0.0), mat);
}

SdfResult sdfBoxNE(vec3 pos, vec3 size, vec3 offset, float mat) {
    pos = rotate(pos-offset);
    // pos = pos-offset;
    vec3 dist = abs(pos) - size;
    return SdfResult(nelength(max(dist, 0.0)) + min(max(dist.x, max(dist.y, dist.z)), 0.0), mat);
}

SdfResult SdfIsocahedron(vec3 p, vec3 offset, float r, float mat){
	float G=sqrt(5.)*.5+.5;
	vec3 n=normalize(vec3(G,1./G,0));
	float d=0.;
    p=sabs(rotate(p-offset), 0.);
    d=smax(d,dot(p,n), 0.);
    d=smax(d,dot(p,n.yzx), 0.);
    d=smax(d,dot(p,n.zxy), 0.);
	d=smax(d,dot(p,normalize(vec3(1))), 0.);
    return SdfResult(d-r, mat);
}

SdfResult SdfIsocahedronNE(vec3 p, vec3 offset, float r, float mat){
	float G=sqrt(5.)*.5+.5;
	vec3 n=normalize(vec3(G,1./G,0.));
	float d=0.;
    p=sabs(rotate(p-offset), 0.);
    // p = p-offset;

    d=smax(d,nedot(p,n), 0.);
    d=smax(d,nedot(p,n.yzx), 0.);
    d=smax(d,nedot(p,n.zxy), 0.);
	d=smax(d,nedot(p,normalize(vec3(1))), 0.);
    // d=dot(p,n)-nedot(p,n);
    // d=0.;
    return SdfResult(d-r, mat);
}


vec2 spherical_map(vec3 p){
    float theta = atan(p.x, p.z);
    float radius = length(p);
    float phi = acos(p.y/radius);
    float raw_u = theta/(2.*PI);
    float u = 1.-(raw_u+0.5);
    float v = 1.-phi/PI;
    return vec2(u,v);
}

SdfResult sdfSphere(vec3 pos, float radius, vec3 offset, float mat) {
    return SdfResult(length(pos - offset) - radius, mat);
}

SdfResult sdfSphereNE(vec3 pos, float radius, vec3 offset, float mat) {
    return SdfResult(nelength(pos - offset) - radius, mat);
}
struct MarchResult {
    float dist;
    float material;
    bool culled;
    int steps;
};

SdfResult combineSmooth(SdfResult d1, SdfResult d2, float r) {
  float m1 = min(d1.dist, d2.dist);
  float m2 = mix(d1.mat, d2.mat, saturate(d1.dist/(d1.dist+d2.dist)));
    // float m2 = d2.mat;
    // float m2 = 1.;

  if (d1.dist < r && d2.dist < r) {
    return SdfResult(min(m1, r - length(r - vec2(d1.dist, d2.dist))), m2);
  } else {
    return SdfResult(m1, m2);
  }
}

SdfResult merge_sdfs(SdfResult[4] sdfs) {
    SdfResult final_sdf = sdfs[0];
    for (int i = 1; i<4; ++i) {
        if (sdfs[i].dist != 0.)
        {
    // int i = 1;
            final_sdf = combineSmooth(final_sdf, sdfs[i], 0.1);
        }
    }
    // return sdfs[3];
    return final_sdf;
}

SdfResult sdfWorld(vec3 p) {
    // float box = 9.;
    SdfResult box = sdfBox(p, vec3(.1, .1, .1), vec3(1., 0.5, 1.), 1.);
    // SdfResult sphere = sdfSphere(p, 0.1, vec3(0.5, 0.5, 1.), 2.);
    SdfResult sphere2 = sdfSphere(p, 0.2, vec3(0.1, 0.5, 1.), 3.);
    // SdfResult sphere2 = sdfSphere(p, 0.1, vec3(0.3, 0.5, 1.), 2.);
    SdfResult plane = sdfPlane(p, vec4(0, 2., 0, 0), -1.);
    SdfResult icosahedron = SdfIsocahedron(p, vec3(0.5, 0.5, 0.5), 0.1, 1.);
    SdfResult null = SdfResult(0., 0.);
    // SdfResult plane = SdfResult(999999999999999., 10.);
    return merge_sdfs(
        SdfResult[4](
            box,
            icosahedron,
            plane,
            sphere2
        )
    );
    // return sphere;
    // return plane;
    // return smin(sphere, plane, 0.);
    // return sdfSphere(p, 0.1, vec3(0.5, 0.5, 2.));
    // return sdfBox(p, vec3(0.1, 0.1, 0.1), vec3(0.5, 0.5, 1.));
    // return sdfBox(p, vec3(100., 10., 100.), vec3(200., 0., 200.));
}


MarchResult rayMarch(vec3 origin, vec3 direction){
    float deltaDepth = 0.;
    float depth = 0.;
    float material = 0.;
    MarchResult result = MarchResult(0., 0., true, 0);

    for(int i = 0; i < max_steps; ++i) {
        SdfResult sdf = sdfWorld(origin + depth*direction);
        deltaDepth = sdf.dist;
        material = sdf.mat;
        depth += deltaDepth;
        if(deltaDepth < epsilon) {
            result.dist = depth;
            result.material = material;
            result.culled = false;
            result.steps = i;
            break;
        }
        if (culling_distance < depth){
            break;
        }
    }
    return result;
}

vec3 calcNormal(vec3 p) {
    // should precalculate all these but eh
    float eps = epsilon;
    return normalize(vec3(
        sdfWorld(p + vec3(eps, 0, 0)).dist - sdfWorld(p + vec3(-eps, 0, 0)).dist,
        sdfWorld(p + vec3(0, eps, 0)).dist - sdfWorld(p + vec3(0, -eps, 0)).dist,
        sdfWorld(p + vec3(0, 0, eps)).dist - sdfWorld(p + vec3(0, 0, -eps)).dist
    ));
}

mat3 computeTBN(vec3 normal, vec3 position, vec2 uv) {
    // Calculate partial derivatives
    vec3 dp1 = dFdx(position);
    vec3 dp2 = dFdy(position);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);

    // Calculate the tangent and bitangent vectors
    vec3 tangent = normalize(dp1 * duv2.y - dp2 * duv1.y);
    vec3 bitangent = normalize(dp2 * duv1.x - dp1 * duv2.x);

    // Return the TBN matrix
    return mat3(tangent, bitangent, normal);
}

vec3 triplanarMap(vec3 surfacePos, vec3 normal, float scale)
{
	// Take projections along 3 axes, sample texture values from each projection, and stack into a matrix
	mat3x3 triMapSamples = mat3x3(
		texture(u_textures[0], surfacePos.yz * scale).rgb,
		texture(u_textures[0], surfacePos.xz * scale).rgb,
		texture(u_textures[0], surfacePos.xy * scale).rgb
		);

	// Weight three samples by absolute value of normal components
	return pow(triMapSamples * abs(normal), vec3(2.2));
    // return triMapSamples * abs(normal);
}

float checkers(vec2 p)
{
    vec2 w = fwidth(p) + 0.001;
    vec2 i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;
    return 0.5 - 0.5*i.x*i.y;
}

float checker(vec2 uv, float repeats) {
  float cx = floor(repeats * uv.x);
  float cy = floor(repeats * uv.y); 
  float result = mod(cx + cy, 2.0);
  return sign(result);
}

mat3 setCamera(in vec3 origin, in vec3 target, float rotation) {
    vec3 forward = normalize(target - origin);
    vec3 orientation = vec3(sin(rotation), cos(rotation), 0.0);
    vec3 left = normalize(cross(forward, orientation));
    vec3 up = normalize(cross(left, forward));
    return mat3(left, up, forward);
}


float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

float D_GGX(float linearRoughness, float NoH, const vec3 h) {
    // Walter et al. 2007, "Microfacet Models for Refraction through Rough Surfaces"
    float oneMinusNoHSquared = 1.0 - NoH * NoH;
    float a = NoH * linearRoughness;
    float k = linearRoughness / (oneMinusNoHSquared + a * a);
    float d = k * k * (1.0 / PI);
    return d;
}

float V_SmithGGXCorrelated(float linearRoughness, float NoV, float NoL) {
    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    float a2 = linearRoughness * linearRoughness;
    float GGXV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
    float GGXL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
    return 0.5 / (GGXV + GGXL);
}

vec3 F_Schlick(const vec3 f0, float VoH) {
    // Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
    return f0 + (vec3(1.0) - f0) * pow5(1.0 - VoH);
}

float F_Schlick(float f0, float f90, float VoH) {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

float Fd_Burley(float linearRoughness, float NoV, float NoL, float LoH) {
    // Burley 2012, "Physically-Based Shading at Disney"
    float f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
    float lightScatter = F_Schlick(1.0, f90, NoL);
    float viewScatter  = F_Schlick(1.0, f90, NoV);
    return lightScatter * viewScatter * (1.0 / PI);
}

float Fd_Lambert() {
    return 1.0 / PI;
}

void main(){
    // todo displacement
   
    // vec2 uv = fragCoord / iResolution.xy;
    // vec2 uv = vec2(u_resolution.x/u_resolution.y*uv.x, uv.y); // aspect ratio fix
    float aspect_ratio = u_resolution.x/u_resolution.y;
    vec2 uv = vec2(uv.x, uv.y/aspect_ratio);
    // vec2 aspect_ratio = vec2(u_resolution.x/u_resolution.y, 1);
    // vec3 sun = vec3(0., 4., 0.); // position of sun
    vec3 sun = normalize(vec3(sin(u_time), 0.9, -0.5));
    // vec3 sun = normalize(vec3(1, 1, 1));

    // Time varying pixel color
    vec3 col = 0.5 + 0.5 * cos(u_time + uv.xyx + vec3(0, 2, 4));
    // vec3 col = vec3(0.);
    // vec3 o = vec3(0.5, 0.5, 0);
    // if (u_mouse.z != 0.){
    //     vec3 o = vec3(u_mouse.xy,0.);
    //     col = vec3(0,0,0);
    // }
    float perspective = 2.;
    // vec3 dir = normalize(vec3(uv.x, 1./perspective, uv.y) - o);
    // vec3 dir = normalize(vec3(uv, 1./perspective)-o);
    vec3 origin = vec3(0.0, 0.8, 0.0);
    vec3 target = vec3(0.1, 0.5, 1.);

    origin.x += 1.7 * cos(u_time * 0.2);
    origin.z += 1.7 * sin(u_time * 0.2);

    mat3 toWorld = setCamera(origin, target, 0.0);
    vec3 direction = toWorld * normalize(vec3(-1. + 2.*uv, 2.0));
    // vec3 p = vec3(uv, 0.);
    MarchResult ray = rayMarch(origin, direction);
    // col = vec3(ray.culled);
    if (!ray.culled ){
        vec3 pos = origin + ray.dist*direction;
        vec3 normal = calcNormal(pos);
        // col = 0.5*normal + 0.5;
        // col = texture(u_textures[0], uv).xyz;
        // if (ray.material > 1.) {
        //     col = vec3(checkers(pos.xz));
        // }
        // m just stores some material identifier, here we're arbitrarily modifying it
        // just to get some different colour values per object
        // col = vec3(0.18*ray.material, 0.6-0.05*ray.material, 0.2);
        float mat = ray.material;

        vec3 plane = vec3(saturate(checkers(pos.xz) + 0.1));
        // offset by location
        vec2 spherical_pos = spherical_map(rotate(pos - vec3(0.1, 0.5, 1.)));
        
        // vec3 sphere2 = vec3(texture(u_textures[0], spherical_pos));

        vec3 col1 = vec3(1., 0., 0.);
        vec3 col2 = vec3(0., 0., 1.);
        // vec3 col3 = vec3(texture(u_textures[0], spherical_map(rotate(pos - vec3(0.1, 0.5, 1.)))));
        
        // col = mix()
        if (-1.5 < mat) {
            col = plane;
        }
        if (0.5 < mat) {
            // col = col2;
            col = mix(col1, col2, saturate(mat-1.));
        }
        if (1.5 < mat) {
            // pos += normal*vec3(texture(u_textures[4], spherical_pos));

            // Compute the TBN matrix
            mat3 TBN = computeTBN(normal, pos, spherical_pos);

            // Sample the normal map
            vec3 normalMap = texture(u_textures[1], spherical_pos).rgb;
            normalMap = normalMap * 2.0 - 1.0; // Transform from [0,1] to [-1,1]

            // Transform the normal from tangent space to world space
            vec3 worldNormal = normalize(TBN * normalMap);

            // Sample the roughness
            vec3 albedo = pow(texture(u_textures[0], spherical_pos).rgb, vec3(2.2));

            //specular shenanigans - straight up stolen
            float roughness = texture(u_textures[2], spherical_pos).r;
            vec3 v = normalize(-direction);
            vec3 l = normalize(sun);
            vec3 h = normalize(v + l);
            vec3 r = normalize(reflect(direction, normal));

            float NoV = abs(dot(normal, v)) + epsilon;
            float NoL = saturate(dot(normal, l));
            float NoH = saturate(dot(normal, h));
            float LoH = saturate(dot(l, h));
            float linearRoughness = roughness * roughness;
            float D = D_GGX(linearRoughness, NoH, h);
            float V = V_SmithGGXCorrelated(linearRoughness, NoV, NoL);
            float metallic = 1.;
            vec3 f0 = 0.04 * (1.0 - metallic) + albedo.rgb * metallic;
            vec3  F = F_Schlick(f0, LoH);
            vec3 Fr = (D * V) * F;

            // diffuse BRDF
            vec3 Fd = albedo * Fd_Burley(linearRoughness, NoV, NoL, LoH);

            float ambient_occlusion = texture(u_textures[3], spherical_pos).r;
            
            col = Fd + Fr;
            col *= ambient_occlusion;
            // col = albedo * ambient_occlusion;
            // col = vec3(V);
                
            // normal = worldNormal;                
            }
        // if (2.5 < mat) {
        //     col = mix(col2, col3, clamp(mat-2., 0., 1.));
        //     // col = plane;
        // }
        // col = vec3(1.,0., 0.);
        // col = vec3(texture(u_textures[0], spherical_map(pos - vec3(0.5, 0.5, 1.0)), 8.));
        // if (length(pos-vec3(0.5, 0., 2.)) < 0.5) {
        //     out_color = vec4(spherical_map(vec3(uv, 1.)), 0., 0.);
        // }

        // col = mat;
        // {
        // // Apply triplanar mapping only to objects with material ID 2
        //     col = triplanarMap(pos, normal, 1.);
        // }
        MarchResult shadowCast = rayMarch(pos + normal*epsilon*1.5, sun);

        col *= max(dot(normal, normalize(sun)), 0.);
        if (!shadowCast.culled) {
            // shadowed
            col = mix(col, col*0.1, 1.);
        }
        
        // col = mix(mix(col1, col2, max(mat-1., 1.)), col3, max(mat-2., 0.));
        // col = mix(mix(col1, col2, max(mat-1., 0.)), col3, max(mat-2., 0.));
        // col = vec3(shadowCast.culled);
        // col = vec3(float(ray.steps)/100., 0, 0.);
    }
    //vec4 q = vec4(0., 0., 0., 0.);
    //vec3 box = vec3(150, 100, 100);
    
    //float delta_depth = sdfWorld(p);
    

    // col = vec3(sdfWorld(p));
    // col = vec3(checker(uv, 8.));
    col = pow(col, vec3(0.4545)); // magic number from gamma curve
    out_color = vec4(col, 1.0);
}
