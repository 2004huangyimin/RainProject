Shader "Custom/PBR_CT" {
    Properties {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _NormalMap ("Normal Map (RGB) Gloss (A)", 2D) = "bump" {}
        _HeightMap ("Height Map", 2D) = "black" {}
        _SkyldrMap ("Environment Cubemap", Cube) = "" {}
        _RippleMap ("Ripple Texture", 2D) = "white" {}
        _ReflectionIntensity ("Reflection Intensity", Range(0, 2)) = 1.0
        _F0 ("Fresnel F0", Range(0, 1)) = 0.04
        _WetLevel ("Base Wet Level", Range(0,1)) = 0.0
        _FloodLevel ("Flood Level", Vector) = (0,0,0,0)
        _RainIntensity ("Rain Intensity", Range(0,1)) = 0.0        
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
            #pragma multi_compile _ SHADOWS_SCREEN
            #pragma multi_compile _ VERTEXLIGHT_ON
            
            #include "UnityCG.cginc"
            #include "AutoLight.cginc"
            #include "Lighting.cginc"
            #include "WetNess.cginc"
            
            struct appdata {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
                float2 texcoord : TEXCOORD0;
                float4 color : COLOR;
            };
            
            struct v2f {
                float4 color : COLOR;
                float2 uv : TEXCOORD0;
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD1;
                float3 tspace0 : TEXCOORD2;
                float3 tspace1 : TEXCOORD3;
                float3 tspace2 : TEXCOORD4;
                SHADOW_COORDS(5) // 阴影坐标
            };
            
            sampler2D _MainTex;
            sampler2D _NormalMap;
            sampler2D _HeightMap;
            sampler2D _RippleMap;
            samplerCUBE _SkyldrMap;
            float _ReflectionIntensity;
            float _F0;
            
            float4 _MainTex_ST;
            float4 _NormalMap_ST;

            float _WetLevel;
            float4 _FloodLevel;
            float _RainIntensity;
            float _AnimationLength;

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
                
                o.color = v.color;//Water Hole height.

                // 传递阴影坐标
                TRANSFER_SHADOW(o);
                
                return o;
            }
            
            fixed4 frag (v2f i) : SV_Target 
            {
                // Timeline or static parameters
                float FloodX = _FloodLevel.x;
                float FloodY = _FloodLevel.y;
                float WetLevel = _WetLevel;
                float RainIntensity = _RainIntensity;

                 if (_AnimationLength > 0)
                {
                    // float AnimTime = fmod(_Time, _AnimationLength);
                    // float4 anim = SAMPLE_TEXTURE2D_LOD(_TimelineTex, sampler_TimelineTex, float2(AnimTime / _AnimationLength, 0.5), 0);
                    // FloodX = anim.z;
                    // FloodY = anim.w;
                    // WetLevel = anim.y;
                    // RainIntensity = anim.x;
                }               

                // 采样纹理
                float Height = tex2D(_HeightMap, i.uv);
                fixed4 albedo = tex2D(_MainTex, i.uv);
                fixed4 normalGloss = tex2D(_NormalMap, i.uv);
                float Gloss = normalGloss.a;
                half3 F0Specular = _F0;                
               
                // 从法线贴图解码法线
                float3 tangentNormal;
                tangentNormal.xyz = normalGloss.rgb * 2 - 1;
                
                // 将法线从切线空间转换到世界空间
                float3 worldNormal;
                worldNormal.x = dot(i.tspace0, tangentNormal);
                worldNormal.y = dot(i.tspace1, tangentNormal);
                worldNormal.z = dot(i.tspace2, tangentNormal);
                worldNormal = normalize(worldNormal);
                
                /////////////////////////////
                // Rain effets - Specific code
                
                // Parameter to customize heightmap for rain if needed
                // because it could not match the one for bumpoffset.                
                float  ScaleHeight = 1.0f;
                float  BiasHeight = 0.0f;
                Height = Height * ScaleHeight + BiasHeight; 

                // Accumulated water
                float2 AccumWater;
                AccumWater.x = min(FloodX, 1.0 - Height);
                AccumWater.y = saturate((FloodY - i.color.g) / 0.4);
                float AccumulatedWater = max(AccumWater.x, AccumWater.y);                

                // Ripple normal
                float3 RippleTangentNormal = tex2D(_RippleMap, i.worldPos.xz * 0.05).rgb * 2 - 1;
                float3 RippleWorldNormal;
                RippleWorldNormal.x = dot(i.tspace0, RippleTangentNormal);
                RippleWorldNormal.y = dot(i.tspace1, RippleTangentNormal);
                RippleWorldNormal.z = dot(i.tspace2, RippleTangentNormal);
                RippleWorldNormal = normalize(RippleWorldNormal);

                float3 WaterNormal = lerp(float3(0,1,0), RippleWorldNormal, saturate(RainIntensity * 100));

                float NewWetLevel = saturate(WetLevel + AccumulatedWater);
                
                // Water influence on material BRDF (no modification of the specular term for now)
                // Type 2 : Wet region
                DoWetProcess(albedo.rgb, Gloss, NewWetLevel);

                // Apply accumulated water effect
                // When AccumulatedWater is 1.0 we are in Type 4
                // so full water properties, in between we are in Type 3
                // Water is smooth
                Gloss = lerp(Gloss, 1.0, AccumulatedWater);
               // Water F0 specular is 0.02 (based on IOR of 1.33)
                F0Specular = lerp(F0Specular, 0.02, AccumulatedWater);
                worldNormal = lerp(worldNormal, WaterNormal, AccumulatedWater);                

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
                float roughness = 1.0 - Gloss;
                roughness = max(roughness, 0.05); // 防止过小的粗糙度
                float3 F0 = F0Specular;
                
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
                float shadow = 1;SHADOW_ATTENUATION(i);
                float3 directLight = _LightColor0.rgb * NdotL * shadow;
                
                // 最终颜色合成
                float3 kS = F;
                float3 kD = (1.0 - kS) * (1.0 - _F0);
                
                float3 directDiffuse = kD * diffuseBRDF * directLight;
                float3 directSpecular = specularBRDF * directLight;
                
                float3 finalColor = directDiffuse + directSpecular + ambientDiffuse + ambientSpecular;
                
                fixed4 col = fixed4(finalColor.xyz, albedo.a);
                return col;
            }
            ENDCG
        }
    }
    //FallBack "Diffuse"
}