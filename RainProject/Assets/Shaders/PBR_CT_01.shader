Shader "Custom/PBR_CT_01"
{
    Properties
    {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _NormalGloss ("Normal (RGB) + Gloss (A)", 2D) = "bump" {}
        _Metallic ("Metallic", Range(0,1)) = 0.0
        _GlossScale ("Gloss Scale", Range(0,1)) = 1.0
        _EnvIntensity ("Environment Intensity", Range(0,5)) = 1.0
        _Exposure ("Exposure", Range(0.01,8)) = 1.0
        _SkyboxTex ("Skybox Cubemap", CUBE) = "" {}
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        Pass
        {
            Tags { "LightMode"="ForwardBase" }
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0

            #include "UnityCG.cginc"
            #include "Lighting.cginc"

            sampler2D _MainTex;
            sampler2D _NormalGloss;
            samplerCUBE _SkyboxTex;

            float4 _MainTex_ST;
            float4 _NormalGloss_ST;

            float _Metallic;
            float _GlossScale;
            float _EnvIntensity;
            float _Exposure;

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uvMain : TEXCOORD0;
                float2 uvNormal : TEXCOORD1;
                float3 worldPos : TEXCOORD2;
                float3 viewDir : TEXCOORD3;
                float3 worldNormal : TEXCOORD4;
                float3 worldTangent : TEXCOORD5;
                float3 worldBitangent : TEXCOORD6;
            };

            float DistributionGGX(float NdotH, float roughness)
            {
                float a = roughness * roughness;
                float a2 = a * a;
                float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
                return a2 / (UNITY_PI * denom * denom + 1e-6);
            }

            float GeometrySchlickGGX(float NdotV, float k)
            {
                return NdotV / (NdotV * (1.0 - k) + k);
            }

            float GeometrySmith(float NdotV, float NdotL, float k)
            {
                float ggx1 = GeometrySchlickGGX(NdotV, k);
                float ggx2 = GeometrySchlickGGX(NdotL, k);
                return ggx1 * ggx2;
            }

            float3 FresnelSchlick(float cosTheta, float3 F0)
            {
                return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
            }

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uvMain = TRANSFORM_TEX(v.uv, _MainTex);
                o.uvNormal = TRANSFORM_TEX(v.uv, _NormalGloss);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;

                float3 n = UnityObjectToWorldNormal(v.normal);
                float3 t = UnityObjectToWorldDir(v.tangent.xyz);
                float3 b = cross(n, t) * v.tangent.w;

                o.worldNormal = normalize(n);
                o.worldTangent = normalize(t);
                o.worldBitangent = normalize(b);


                float3 viewPos = _WorldSpaceCameraPos;
                o.viewDir = normalize(viewPos - o.worldPos);

                return o;
            }

            fixed4 frag(v2f IN) : SV_Target
            {
                fixed4 albedoS = tex2D(_MainTex, IN.uvMain);
                fixed4 normalGlossS = tex2D(_NormalGloss, IN.uvNormal);

                float3 n_t = normalGlossS.xyz * 2.0 - 1.0;
                float gloss = saturate(normalGlossS.a);
                float smoothness = saturate(gloss * _GlossScale);
                float roughness = max(0.001, 1.0 - smoothness);

                float3 T = normalize(IN.worldTangent);
                float3 B = normalize(IN.worldBitangent);
                float3 N = normalize(IN.worldNormal);
                float3x3 TBN = float3x3(T, B, N);
                float3 worldNormal = normalize(mul(TBN, n_t));

                float3 V = normalize(IN.viewDir);
                float3 L = normalize(_WorldSpaceLightPos0.xyz);
                float3 radiance = _LightColor0.rgb;

                float3 H = normalize(V + L);
                float NdotL = saturate(dot(worldNormal, L));
                float NdotV = saturate(dot(worldNormal, V));
                float NdotH = saturate(dot(worldNormal, H));
                float VdotH = saturate(dot(V, H));

                float3 baseColor = albedoS.rgb;
                float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, _Metallic);

                float D = DistributionGGX(NdotH, roughness);
                float k = (roughness + 1.0);
                k = (k * k) / 8.0;
                float G = GeometrySmith(NdotV, NdotL, k);
                float3 F = FresnelSchlick(VdotH, F0);

                float3 numerator = D * G * F;
                float denom = 4.0 * max(NdotV * NdotL, 1e-6);
                float3 specular = numerator / denom;

                float3 kD = (1.0 - F) * (1.0 - _Metallic);
                float3 diffuse = kD * baseColor / UNITY_PI;

                float3 Lo = (diffuse + specular) * radiance * NdotL;

                float3 R = reflect(-V, worldNormal);
                float3 envColor = texCUBE(_SkyboxTex, R).rgb * _EnvIntensity;
                float3 irradiance = texCUBE(_SkyboxTex, worldNormal).rgb * _EnvIntensity;

                float3 diffuseIBL = kD * baseColor * irradiance / UNITY_PI;
                float3 specularIBL = envColor * F;
                float3 ambient = diffuseIBL + specularIBL;

                float3 colorLinear = Lo + ambient * _Exposure;

                return float4(IN.worldBitangent, 1.0);
            }
            ENDCG
        }
    }

    FallBack "Diffuse"
}
