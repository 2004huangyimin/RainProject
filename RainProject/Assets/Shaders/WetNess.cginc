#define PI 3.141592653
sampler2D _RippleTexture;
sampler2D _DynamicRippleTexture;

void DoWetProcess(inout float3 Diffuse, inout float Gloss, float WetLevel)
{
   // Water influence on material BRDF
   Diffuse    *= lerp(1.0, 0.3, WetLevel);                   // Attenuate diffuse
   Gloss       = min(Gloss * lerp(1.0, 2.5, WetLevel), 1.0); // Boost gloss
}

// Compute a ripple layer for the current time
float3 ComputeRipple(float2 UV, float CurrentTime, float Weight)
{
   float4 Ripple = tex2D(_RippleTexture, UV);
   Ripple.yz = Ripple.yz * 2.0 - 1.0;
            
   float DropFrac = frac(Ripple.w + CurrentTime);
   float TimeFrac = DropFrac - 1.0 + Ripple.x;
   float DropFactor = saturate(0.2 + Weight * 0.8 - DropFrac);
   float FinalFactor = DropFactor * Ripple.x * sin( clamp(TimeFrac * 9.0, 0.0f, 3.0) * PI);
   
   return float3(Ripple.yz * FinalFactor * 0.35, 1.0);
}