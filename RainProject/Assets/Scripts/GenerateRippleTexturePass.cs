using System.Collections;
using System.Collections.Generic;
using JetBrains.Annotations;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

[ExecuteInEditMode]
public class GenerateRippleTexturePass : MonoBehaviour
{
    public Texture2D RippleTexture;
    public Material matGenerateRipple;
    private RenderTexture rtRippleMap = null;
    private CommandBuffer cmd = null;
    private Camera Cam = null;
    private const int TEXTURE_SIZE = 256;
    private const string RIPPLE_PASS_NAME = "GenerateRippTexturePass";
    private const string DYNAMIC_RIPPLE_TEXTURE_NAME = "DynamicRippleTexture";
    // Start is called before the first frame update
    void Awake()
    {
        cmd = new CommandBuffer();
        cmd.name = RIPPLE_PASS_NAME;
        Cam = GetComponent<Camera>();
        if (Cam != null)
        {
            if (Cam.commandBufferCount > 0)
            {
                Cam.RemoveAllCommandBuffers();
            }
            Cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, cmd);
        }
        if (rtRippleMap == null)
        {
            rtRippleMap = new RenderTexture(TEXTURE_SIZE, TEXTURE_SIZE, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            rtRippleMap.filterMode = FilterMode.Bilinear;
            rtRippleMap.wrapMode = TextureWrapMode.Repeat;
            rtRippleMap.name = DYNAMIC_RIPPLE_TEXTURE_NAME;
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (matGenerateRipple != null && Cam != null && cmd != null && rtRippleMap != null && rtRippleMap.width > 0)
        {
            cmd.Clear();
            cmd.Blit(RippleTexture, rtRippleMap, matGenerateRipple);
            cmd.SetGlobalTexture(ComonData.DYNAMIC_RIPPLE_TEXTURE, rtRippleMap);
        }
    }
}
