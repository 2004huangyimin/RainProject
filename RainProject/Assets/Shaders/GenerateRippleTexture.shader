Shader "Custom/GenerateRippleTexture"
{
    Properties
    {
        _RippleTexture ("Ripple Texture", 2D) = "white" {}
        _RainIntensity ("Rain Intensity", Range(0,1)) = 0.5        
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            #include "WetNess.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float _CustomTime; // custom time
            float _RainIntensity;

            fixed4 frag (v2f i) : SV_Target
            {
                float RainIntensity = _RainIntensity;

                float4 TimeMul = float4(1.0, 0.85, 0.93, 1.13);
                float4 TimeAdd = float4(0.0, 0.2, 0.45, 0.7);
                float GlobalMul = 1.6;
                float4 Times = frac((_Time.y * TimeMul + TimeAdd) * GlobalMul);

                float2 UV = i.uv;
                float4 Weights = saturate((RainIntensity - float4(0,0.25,0.5,0.75)) * 4);

                float3 Ripple1 = ComputeRipple(UV + float2( 0.25, 0.0 ), Times.x, Weights.x);
                float3 Ripple2 = ComputeRipple(UV + float2(-0.55, 0.3 ), Times.y, Weights.y);
                float3 Ripple3 = ComputeRipple(UV + float2( 0.60, 0.85), Times.z, Weights.z);
                float3 Ripple4 = ComputeRipple(UV + float2( 0.50,-0.75), Times.w, Weights.w);

                float4 Z = lerp(1.0, float4(Ripple1.z, Ripple2.z, Ripple3.z, Ripple4.z), Weights);
                float3 Normal = float3(
                    Weights.x * Ripple1.xy +
                    Weights.y * Ripple2.xy +
                    Weights.z * Ripple3.xy +
                    Weights.w * Ripple4.xy,
                    Z.x * Z.y * Z.z * Z.w
                );

                return float4(normalize(Normal) * 0.5 + 0.5, 1.0);
            }
            ENDCG
        }
    }
}
