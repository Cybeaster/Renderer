#shader vertex
#version 430


layout(location=0) in vec3 vertPosition;
layout(location=1) in vec2 texels;
layout(location=2) in vec3 normal;

out vec3 varyingNormal;

#define MAX_POINTS_OF_LIGHT 5
out vec3 varyingSpotLightDirections[MAX_POINTS_OF_LIGHT];
out vec3 varyingSpotHalfVectors[MAX_POINTS_OF_LIGHT];

out vec3 varyingPointLightDirections[MAX_POINTS_OF_LIGHT];
out vec3 varyingPointHalfVectors[MAX_POINTS_OF_LIGHT];

out vec3 varyingVertPos;
out vec4 shadowCoord;

struct AttenuationFactor
{
    float constant;
    float quadratic;
    float linear;
};

struct LightBase
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

struct DirectionalLight
{
    LightBase base;
    vec3 direction;
};

struct PointLight
{
    LightBase base;
    vec3 position;
    AttenuationFactor attenuation;
};

struct Spotlight
{
    PointLight base;
    vec3 direction;
    float cutoff;
};

struct Material
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
};


uniform Spotlight spotLights[MAX_POINTS_OF_LIGHT];
uniform PointLight pointLights[MAX_POINTS_OF_LIGHT];

uniform uint numPointLights;
uniform uint numSpotLights;

uniform vec4 globalAmbient;
uniform Material material;
uniform mat4 projMatrix;
uniform mat4 mvMatrix;
uniform mat4 normMatrix;

uniform float windowHeight;
uniform float windowWidth;

uniform int useTexture = 1;

uniform mat4 shadowMVP;

layout (binding=0) uniform sampler2DShadow shadowTexture;

void main(void)
{
    varyingVertPos = (mvMatrix * vec4(vertPosition, 1.0)).xyz;

    for (int i =0; i < numSpotLights; i++)
    {
        varyingSpotLightDirections[i] = spotLights[i].base.position - varyingVertPos;
        varyingSpotHalfVectors[i] = (varyingSpotLightDirections[i] + (-varyingVertPos)).xyz;
    }

    for (int i = 0; i < numPointLights;i++)
    {
        varyingPointLightDirections[i] = pointLights[i].position - varyingVertPos;
        varyingPointHalfVectors[i] = (varyingPointLightDirections[i] + (-varyingVertPos)).xyz;
    }

    shadowCoord = shadowMVP * vec4(vertPosition, 1.0);

    varyingNormal = (normMatrix * vec4(normal, 1.0)).xyz;
    gl_Position = projMatrix * mvMatrix * vec4(vertPosition, 1.0);
}




#shader fragment
#version 430


#define MAX_POINTS_OF_LIGHT 5

in vec3 varyingSpotLightDirections[MAX_POINTS_OF_LIGHT];
in vec3 varyingSpotHalfVectors[MAX_POINTS_OF_LIGHT];
in vec3 varyingPointLightDirections[MAX_POINTS_OF_LIGHT];
in vec3 varyingPointHalfVectors[MAX_POINTS_OF_LIGHT];

in vec3 varyingNormal;
in vec3 varyingVertPos;
in vec4 shadowCoord;

out vec4 fragColor;

layout (binding = 0) uniform sampler2DShadow shadowTexture;


struct AttenuationFactor
{
    float constant;
    float quadratic;
    float linear;
};

struct LightBase
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

struct DirectionalLight
{
    LightBase base;
    vec3 direction;
};

struct PointLight
{
    LightBase base;
    vec3 position;
    AttenuationFactor attenuation;
};

struct Spotlight
{
    PointLight base;
    vec3 direction;
    float cutoff;
};

struct Material
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
};


uniform uint numPointLights;
uniform uint numSpotLights;

uniform Spotlight spotLights[MAX_POINTS_OF_LIGHT];
uniform PointLight pointLights[MAX_POINTS_OF_LIGHT];

uniform vec4 globalAmbient;
uniform Material material;
uniform mat4 projMatrix;
uniform mat4 mvMatrix;
uniform mat4 normMatrix;
uniform int useTexture = 1;

uniform float windowHeight;
uniform float windowWidth;

float ShadowLookUp(float Ox, float Oy)
{
    return textureProj(shadowTexture, shadowCoord + vec4(Ox * (1/windowWidth) * shadowCoord.w, Oy * (1/windowHeight) * shadowCoord.w, -0.01, 0.0));
}

vec4 CalcLightInternal(LightBase Light, vec3 Direction, vec3 Normal, vec3 VaryingHalfVector)
{
    vec3 L = normalize(Direction);
    vec3 N = normalize(Normal);
    vec3 H = normalize(VaryingHalfVector);

    float diffuseFactor = max(dot(L, N), 0.0);// perfomance update
    float specularFactor = max(dot(H, N), 0.0);

    vec4 ambientColor = ((globalAmbient * material.ambient) + (Light.ambient * material.ambient));
    vec4 diffuse = Light.diffuse * material.diffuse * diffuseFactor;
    vec4 specular = Light.specular * material.specular * pow(specularFactor, material.shininess * 3.0);

    float swidth = 2.5;

    float shadowFactor = 0.0;
    vec2 offset = mod(floor(gl_FragCoord.xy), 2.0) * swidth;
    shadowFactor += ShadowLookUp(-1.5*swidth + offset.x, 1.5 * swidth - offset.y);
    shadowFactor += ShadowLookUp(-1.5*swidth + offset.x, -0.5 * swidth - offset.y);
    shadowFactor += ShadowLookUp(0.5 *swidth + offset.x, 1.5 * swidth - offset.y);
    shadowFactor += ShadowLookUp(0.5 *swidth + offset.x, -0.5 * swidth - offset.y);
    shadowFactor /= 4.0;

    //    float endup = swidth * 3.0 + swidth / 2.0;
    //    for(float m = -endup; m <= endup; m += swidth)
    //    {
    //        for(float n = -endup; n <= endup; n+=swidth)
    //        {
    //            shadowFactor += ShadowLookUp(m,n);
    //        }
    //    }
    //    shadowFactor /=64.0;

    return ambientColor + (diffuse + specular) * shadowFactor;
}


vec4 CalcPointLight(PointLight Light, vec3 Normal, vec3 VaryingDir, vec3 HalfVector)
{
    vec3 dir = varyingVertPos - Light.position;
    float distance = length(dir);

    float attenuation = (Light.attenuation.constant + Light.attenuation.linear * distance + Light.attenuation.quadratic * (distance * distance));
    vec4 color = CalcLightInternal(Light.base, VaryingDir, Normal, HalfVector);
    return color / attenuation;
}

vec4 CalcDirectionalLight(DirectionalLight Light, vec3 Normal, vec3 HalfVector)
{
    return CalcLightInternal(Light.base, Light.direction, Normal, HalfVector);
}

vec4 CalcSpotLight(Spotlight Light, vec3 Normal, vec3 LightDir, vec3 HalfVector)
{
    vec3 lightToPixel = normalize(varyingVertPos -  Light.base.position);
    float spotFactor = dot(lightToPixel, Light.direction);

    vec4 color = CalcPointLight(Light.base, Normal, LightDir, HalfVector);
    float spotLightIntensity = (1.0 - (1.0 - spotFactor) / (1.0 - Light.cutoff));
    return color * (spotLightIntensity + globalAmbient);

}


void main(void)
{
    fragColor = vec4(0, 0, 0, 0);

    vec4 pointLightsContribution = vec4(0, 0, 0, 0);
    for (int i = 0; i < numPointLights; i++)
    {
        vec3 halfVector = varyingPointHalfVectors[i];
        vec3 varyingDir = varyingPointLightDirections[i];
        pointLightsContribution = pointLightsContribution + CalcPointLight(pointLights[i], varyingNormal, varyingDir, halfVector) / numPointLights;
    }

    vec4 spotLightContribution = vec4(0, 0, 0, 0);
    for (int i = 0; i < numSpotLights; i++)
    {
        vec3 halfVector = varyingSpotHalfVectors[i];
        vec3 varyingDir = varyingSpotLightDirections[i];
        spotLightContribution = pointLightsContribution + CalcSpotLight(spotLights[i], varyingNormal, varyingDir, halfVector) / numSpotLights;
    }

    fragColor = pointLightsContribution + spotLightContribution;
}