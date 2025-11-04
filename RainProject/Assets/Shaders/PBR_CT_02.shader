Shader "Custom/PBR_CT_02" {
    Properties {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _NormalMap ("Normal Map (RGB) Gloss (A)", 2D) = "bump" {}
        _SkyldrMap ("Environment Cubemap", Cube) = "" {}
        _ReflectionIntensity ("Reflection Intensity", Range(0, 2)) = 1.0
        _F0 ("Fresnel F0", Range(0, 1)) = 0.04
    }
    
    SubShader {
        Tags { "RenderType"="Opaque" }
        LOD 200
        
        Pass {
            Name "FORWARD"
            Tags { "LightMode" = "ForwardBase" }
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fwdbase
            #pragma multi_compile_fog
            
            #include "UnityCG.cginc"
            #include "AutoLight.cginc"
            #include "Lighting.cginc"
            
            struct appdata {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
                float2 texcoord : TEXCOORD0;
            };
            
            struct v2f {
                float2 uv : TEXCOORD0;
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD1;
                float3 tspace0 : TEXCOORD2;
                float3 tspace1 : TEXCOORD3;
                float3 tspace2 : TEXCOORD4;
                UNITY_FOG_COORDS(5)
                LIGHTING_COORDS(6,7)
            };
            
            sampler2D _MainTex;
            sampler2D _NormalMap;
            samplerCUBE _SkyldrMap;
            float _ReflectionIntensity;
            float _F0;
            
            float4 _MainTex_ST;
            
            // GGX/Trowbridge-Reitz 法线分布函数
            float D_GGX(float NdotH, float roughness) {
                float a = roughness * roughness;
                float a2 = a * a;
                float NdotH2 = NdotH * NdotH;
                float denom = (NdotH2 * (a2 - 1.0) + 1.0);
                denom = UNITY_PI * denom * denom;
                return a2 / max(denom, 0.0000001);
            }
            
            // Schlick-Beckmann 几何遮蔽函数
            float G_SchlickBeckmann(float NdotV, float roughness) {
                float k = (roughness * roughness) / 2.0;
                return NdotV / (NdotV * (1.0 - k) + k);
            }
            
            // Smith 几何遮蔽函数
            float G_Smith(float NdotV, float NdotL, float roughness) {
                return G_SchlickBeckmann(NdotV, roughness) * G_SchlickBeckmann(NdotL, roughness);
            }
            
            // Fresnel-Schlick 近似
            float3 F_Schlick(float cosTheta, float3 F0) {
                return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
            }
            
            v2f vert (appdata v) {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                
                // 计算切线空间矩阵
                float3 worldNormal = normalize(mul(v.normal, (float3x3)unity_WorldToObject));
                float3 worldTangent = normalize(mul((float3x3)unity_ObjectToWorld, v.tangent.xyz));
                float3 worldBinormal = cross(worldNormal, worldTangent) * v.tangent.w;
                
                o.tspace0 = float3(worldTangent.x, worldBinormal.x, worldNormal.x);
                o.tspace1 = float3(worldTangent.y, worldBinormal.y, worldNormal.y);
                o.tspace2 = float3(worldTangent.z, worldBinormal.z, worldNormal.z);
                
                UNITY_TRANSFER_FOG(o,o.pos);
                TRANSFER_VERTEX_TO_FRAGMENT(o);
                
                return o;
            }
            
            fixed4 frag (v2f i) : SV_Target {
                // 采样纹理
                fixed4 albedo = tex2D(_MainTex, i.uv);
                fixed4 normalGloss = tex2D(_NormalMap, i.uv);
                
                // 从法线贴图解码法线
                float3 tangentNormal;
                tangentNormal.xyz = normalGloss.rgb * 2 - 1;
                
                // 将法线从切线空间转换到世界空间
                float3 worldNormal;
                worldNormal.x = dot(i.tspace0, tangentNormal);
                worldNormal.y = dot(i.tspace1, tangentNormal);
                worldNormal.z = dot(i.tspace2, tangentNormal);
                worldNormal = normalize(worldNormal);
                
                // 基础向量
                float3 N = worldNormal;
                float3 V = normalize(_WorldSpaceCameraPos.xyz - i.worldPos);
                float3 L = normalize(_WorldSpaceLightPos0.xyz);
                float3 H = normalize(V + L);
                float3 R = reflect(-V, N);
                
                // 点积
                float NdotL = saturate(dot(N, L));
                float NdotV = saturate(dot(N, V));
                float NdotH = saturate(dot(N, H));
                float VdotH = saturate(dot(V, H));
                
                // PBR参数
                float gloss = normalGloss.a;
                float roughness = 1.0 - gloss;
                roughness = max(roughness, 0.05); // 防止过小的粗糙度
                float3 F0 = _F0;
                
                // 计算BRDF分量
                float D = D_GGX(NdotH, roughness);
                float G = G_Smith(NdotV, NdotL, roughness);
                float3 F = F_Schlick(VdotH, F0);
                
                // Cook-Torrance BRDF
                float3 numerator = D * G * F;
                float denominator = 4.0 * NdotV * NdotL;
                float3 specularBRDF = numerator / max(denominator, 0.0000001);
                
                // 漫反射项 (Lambert)
                float3 diffuseBRDF = albedo.rgb / UNITY_PI;
                
                // 环境反射
                float3 reflection = texCUBE(_SkyldrMap, R).rgb;
                float3 ambientSpecular = reflection * F_Schlick(NdotV, F0) * _ReflectionIntensity;
                
                // 环境漫反射
                float3 ambientDiffuse = albedo.rgb * UNITY_LIGHTMODEL_AMBIENT.rgb;
                
                // 直接光照
                float attenuation = LIGHT_ATTENUATION(i);
                float3 directLight = _LightColor0.rgb * NdotL * attenuation;
                
                // 最终颜色合成
                float3 kS = F;
                float3 kD = (1.0 - kS) * (1.0 - _F0);
                
                float3 directDiffuse = kD * diffuseBRDF * directLight;
                float3 directSpecular = specularBRDF * directLight;
                
                float3 finalColor = directDiffuse + directSpecular + ambientDiffuse + ambientSpecular;
                
                fixed4 col = fixed4(finalColor.xyz, albedo.a);
                UNITY_APPLY_FOG(i.fogCoord, col);
                return col;
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
}