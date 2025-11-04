Shader "Custom/PBR_CT_00"
{
    Properties
    {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _BumpGlossMap ("Normal (RGB) + Gloss (A)", 2D) = "bump" {}
        _NormalScale ("Normal Scale", Range(0,4)) = 1
        _F0Color ("Specular F0", Color) = (0.04, 0.04, 0.04, 1)
        _SkyLDR ("Sky LDR Cubemap", Cube) = "_Skybox" {}
        _IBLIntensity ("IBL Intensity", Range(0,4)) = 1
        _EnvDiffuseIntensity ("Env Diffuse Intensity", Range(0,4)) = 1
        _EnvMipCount ("Env Mip Count (manual)", Range(0,10)) = 7
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 300

        Pass
        {
            Tags { "LightMode" = "ForwardBase" }

            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fwdbase
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
            #include "AutoLight.cginc"
            #include "UnityLightingCommon.cginc"

            sampler2D _MainTex;
            float4 _MainTex_ST;
            sampler2D _BumpGlossMap;
            float4 _BumpGlossMap_ST;
            float _NormalScale;
            float4 _F0Color; // F0

            samplerCUBE _SkyLDR;
            float _IBLIntensity;
            float _EnvDiffuseIntensity;
            float _EnvMipCount; // manual LOD range for specular prefilter approximation

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
                float2 uv0 : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                float3 worldNormal : TEXCOORD2;
                float3 worldTangent : TEXCOORD3;
                float3 worldBitangent : TEXCOORD4;
                UNITY_FOG_COORDS(5)
                SHADOW_COORDS(6)
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv0 = TRANSFORM_TEX(v.uv, _MainTex);

                float3 wPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                float3 wNormal = UnityObjectToWorldNormal(v.normal);
                float3 wTangent = UnityObjectToWorldDir(v.tangent.xyz);
                float3 wBitangent = cross(wNormal, wTangent) * v.tangent.w;

                o.worldPos = wPos;
                o.worldNormal = wNormal;
                o.worldTangent = wTangent;
                o.worldBitangent = wBitangent;

                UNITY_TRANSFER_FOG(o, o.pos);
                TRANSFER_SHADOW(o);
                return o;
            }

            // Helpers
            float3x3 BuildTBN(float3 t, float3 b, float3 n)
            {
                return float3x3(t, b, n);
            }

            // Schlick Fresnel
            float3 FresnelSchlick(float cosTheta, float3 F0)
            {
                return F0 + (1.0 - F0) * pow(1.0 - saturate(cosTheta), 5.0);
            }

            // GGX / Trowbridge-Reitz NDF
            float D_GGX(float NdotH, float alpha)
            {
                float a2 = alpha * alpha;
                float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
                return a2 / (UNITY_PI * d * d + 1e-7);
            }

            // Smith GGX visibility using Schlick-G approximation
            float G_Smith_SchlickGGX(float NdotV, float NdotL, float alpha)
            {
                float r = alpha + 1.0;
                float k = (r * r) / 8.0; // UE4 k for direct lighting
                float gv = NdotV / (NdotV * (1.0 - k) + k);
                float gl = NdotL / (NdotL * (1.0 - k) + k);
                return gv * gl;
            }

            float3 SampleEnvSpecular(float3 R, float roughness)
            {
                // Map roughness [0,1] to mip [0, _EnvMipCount]
                float lod = roughness * _EnvMipCount;
                #if defined(UNITY_NO_SCREENSPACE_SHADOWS)
                    // use textureLod path when available
                    #if defined(SHADER_API_D3D11) || defined(SHADER_API_GLCORE) || defined(SHADER_API_METAL) || defined(SHADER_API_VULKAN)
                        return texCUBElod(_SkyLDR, float4(R, lod)).rgb;
                    #else
                        return texCUBE(_SkyLDR, R).rgb; // fallback
                    #endif
                #else
                    #if defined(SHADER_API_D3D11) || defined(SHADER_API_GLCORE) || defined(SHADER_API_METAL) || defined(SHADER_API_VULKAN)
                        return texCUBElod(_SkyLDR, float4(R, lod)).rgb;
                    #else
                        return texCUBE(_SkyLDR, R).rgb;
                    #endif
                #endif
            }

            float3 SampleEnvDiffuse(float3 N)
            {
                // Use the highest mip as a cheap irradiance approximation
                float lod = _EnvMipCount;
                #if defined(SHADER_API_D3D11) || defined(SHADER_API_GLCORE) || defined(SHADER_API_METAL) || defined(SHADER_API_VULKAN)
                    return texCUBElod(_SkyLDR, float4(N, lod)).rgb;
                #else
                    return texCUBE(_SkyLDR, N).rgb;
                #endif
            }

            float3 UnpackNormalRGB_WithScale(float3 enc, float scale)
            {
                // Normal stored in RGB as tangent-space vector
                float3 n = enc * 2.0 - 1.0;
                n.xy *= scale;
                return normalize(n);
            }

            struct FragOut { float4 color : SV_Target; };

            FragOut frag (v2f i)
            {
                FragOut o;

                // Fetch textures
                float2 uvMain = i.uv0;
                float2 uvNG = TRANSFORM_TEX(i.uv0, _BumpGlossMap);
                float4 albedoSample = tex2D(_MainTex, uvMain);
                float4 ngSample = tex2D(_BumpGlossMap, uvNG);

                float3 albedo = albedoSample.rgb; // linear in linear color space

                // Normal map RG, gloss in A
                float gloss = ngSample.a; // [0..1] smoothness
                float roughness = saturate(1.0 - gloss);
                float alphaR = max(1e-3, roughness * roughness);

                // Build TBN and transform normal to world
                float3x3 TBN = BuildTBN(normalize(i.worldTangent), normalize(i.worldBitangent), normalize(i.worldNormal));
                float3 nTS = UnpackNormalRGB_WithScale(ngSample.rgb, _NormalScale);
                float3 N = normalize(mul(TBN, nTS));

                float3 V = normalize(_WorldSpaceCameraPos.xyz - i.worldPos);
                float3 L = normalize(_WorldSpaceLightPos0.xyz);
                float3 H = normalize(V + L);

                float NdotL = saturate(dot(N, L));
                float NdotV = saturate(dot(N, V));
                float NdotH = saturate(dot(N, H));
                float VdotH = saturate(dot(V, H));

                // Cook-Torrance terms
                float D = D_GGX(NdotH, alphaR);
                float G = G_Smith_SchlickGGX(NdotV, NdotL, alphaR);
                float3 F0 = _F0Color.rgb;
                float3 F = FresnelSchlick(VdotH, F0);

                float3 specularBRDF = (D * G) * F / max(4.0 * NdotL * NdotV, 1e-4);
                float3 kd = (1.0.xxx - F); // energy conservation (non-metal workflow)

                // Direct lighting
                UNITY_LIGHT_ATTENUATION(att, i, i.worldPos);
                float3 directDiffuse = (albedo / UNITY_PI) * NdotL * att;
                float3 directSpecular = specularBRDF * NdotL * att;

                // Image-based lighting
                float3 R = reflect(-V, N);
                float3 envSpec = SampleEnvSpecular(R, roughness) * _IBLIntensity;
                float3 envDiffuse = SampleEnvDiffuse(N) * _EnvDiffuseIntensity;

                // Split-sum approximation: approximate specular IBL by multiplying with Fresnel at NdotV
                float3 F_env = FresnelSchlick(NdotV, F0);
                float3 specIBL = envSpec * F_env;
                float3 diffIBL = envDiffuse * albedo / UNITY_PI;

                float3 color = kd * (directDiffuse + diffIBL) + (directSpecular + specIBL);

                // Fog
                UNITY_APPLY_FOG(i.fogCoord, color);
                o.color = float4(color, 1.0);
                return o;
            }
            ENDCG
        }

        // Additive per-pixel lights
        Pass
        {
            Tags { "LightMode" = "ForwardAdd" }
            Blend One One
            Fog { Color (0,0,0,0) }

            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fwdadd

            #include "UnityCG.cginc"
            #include "AutoLight.cginc"

            sampler2D _MainTex; float4 _MainTex_ST;
            sampler2D _BumpGlossMap; float4 _BumpGlossMap_ST; float _NormalScale;
            float4 _F0Color;

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
                float2 uv0 : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                float3 worldNormal : TEXCOORD2;
                float3 worldTangent : TEXCOORD3;
                float3 worldBitangent : TEXCOORD4;
                LIGHTING_COORDS(5,6)
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv0 = TRANSFORM_TEX(v.uv, _MainTex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                o.worldTangent = UnityObjectToWorldDir(v.tangent.xyz);
                o.worldBitangent = cross(o.worldNormal, o.worldTangent) * v.tangent.w;
                TRANSFER_VERTEX_TO_FRAGMENT(o);
                return o;
            }

            float3x3 BuildTBN(float3 t, float3 b, float3 n) { return float3x3(t,b,n); }
            float3 UnpackNormalRGB_WithScale(float3 enc, float scale)
            {
                float3 n = enc * 2.0 - 1.0; n.xy *= scale; return normalize(n);
            }
            float D_GGX(float NdotH, float a) { float a2=a*a; float d=(NdotH*NdotH)*(a2-1.0)+1.0; return a2/(UNITY_PI*d*d+1e-7); }
            float G_Smith_SchlickGGX(float NdotV, float NdotL, float a){ float r=a+1.0; float k=(r*r)/8.0; float gv=NdotV/(NdotV*(1.0-k)+k); float gl=NdotL/(NdotL*(1.0-k)+k); return gv*gl; }
            float3 FresnelSchlick(float cosT, float3 F0){ return F0 + (1.0 - F0) * pow(1.0 - saturate(cosT), 5.0); }

            float4 frag (v2f i) : SV_Target
            {
                float4 albedoSample = tex2D(_MainTex, i.uv0);
                float4 ngSample = tex2D(_BumpGlossMap, TRANSFORM_TEX(i.uv0, _BumpGlossMap));
                float3 albedo = albedoSample.rgb;
                float gloss = ngSample.a;
                float roughness = saturate(1.0 - gloss);
                float alphaR = max(1e-3, roughness * roughness);

                float3x3 TBN = BuildTBN(normalize(i.worldTangent), normalize(i.worldBitangent), normalize(i.worldNormal));
                float3 nTS = UnpackNormalRGB_WithScale(ngSample.rgb, _NormalScale);
                float3 N = normalize(mul(TBN, nTS));

                float3 L = normalize(_WorldSpaceLightPos0.xyz);
                float3 V = normalize(_WorldSpaceCameraPos.xyz - i.worldPos);
                float3 H = normalize(V + L);
                float NdotL = saturate(dot(N,L));
                float NdotV = saturate(dot(N,V));
                float NdotH = saturate(dot(N,H));
                float VdotH = saturate(dot(V,H));

                float D = D_GGX(NdotH, alphaR);
                float G = G_Smith_SchlickGGX(NdotV, NdotL, alphaR);
                float3 F = FresnelSchlick(VdotH, _F0Color.rgb);
                float3 spec = (D*G)*F / max(4.0*NdotL*NdotV, 1e-4);
                float3 kd = (1.0.xxx - F);

                float atten = LIGHT_ATTENUATION(i);
                float3 color = kd * (albedo/UNITY_PI) * NdotL * atten + spec * NdotL * atten;
                return float4(color, 1.0);
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
}


