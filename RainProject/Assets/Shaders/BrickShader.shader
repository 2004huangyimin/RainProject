Shader "Customized/BrickShader"
{
    Properties {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _BumpMap ("Normal Map", 2D) = "bump" {}
        _BumpScale ("Normal Scale", Float) = 1.0
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
            #include "UnityCG.cginc"
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
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            sampler2D _BumpMap;
            float _BumpScale;

            v2f vert (appdata v) {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                
                // 构建切线空间矩阵
                float3 worldNormal = UnityObjectToWorldNormal(v.normal);
                float3 worldTangent = UnityObjectToWorldDir(v.tangent.xyz);
                float3 worldBinormal = cross(worldNormal, worldTangent) * v.tangent.w;
                o.tspace0 = float3(worldTangent.x, worldBinormal.x, worldNormal.x);
                o.tspace1 = float3(worldTangent.y, worldBinormal.y, worldNormal.y);
                o.tspace2 = float3(worldTangent.z, worldBinormal.z, worldNormal.z);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target {
                // 从法线贴图获取切线空间法线
                float3 tangentNormal = UnpackNormal(tex2D(_BumpMap, i.uv));
                tangentNormal.xy *= _BumpScale;
                tangentNormal.z = sqrt(1.0 - saturate(dot(tangentNormal.xy, tangentNormal.xy)));
                
                // 将法线转换到世界空间
                float3 worldNormal;
                worldNormal.x = dot(i.tspace0, tangentNormal);
                worldNormal.y = dot(i.tspace1, tangentNormal);
                worldNormal.z = dot(i.tspace2, tangentNormal);
                
                // 光照计算
                float3 lightDir = normalize(UnityWorldSpaceLightDir(i.worldPos));
                float3 albedo = tex2D(_MainTex, i.uv).rgb;
                
                // 半兰伯特光照模型
                float NdotL = dot(worldNormal, lightDir);
                float halfLambert = NdotL * 0.5 + 0.5;
                float3 diffuse = _LightColor0.rgb * albedo * halfLambert;
                
                // 添加环境光
                float3 ambient = ShadeSH9(float4(worldNormal, 1));
                float3 color = diffuse + ambient * albedo;
                
                return fixed4(color, 1.0);
            }
            ENDCG
        }
    }    
}
