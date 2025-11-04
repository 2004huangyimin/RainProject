void DoWetProcess(inout float3 Diffuse, inout float Gloss, float WetLevel)
{
   // Water influence on material BRDF
   Diffuse    *= lerp(1.0, 0.3, WetLevel);                   // Attenuate diffuse
   Gloss       = min(Gloss * lerp(1.0, 2.5, WetLevel), 1.0); // Boost gloss
}