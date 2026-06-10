using UnityEngine;

public class FrustumMeshGenerator : MonoBehaviour
{
    public Material meshMaterial;
    public float frustumLength = 10f;

    private MeshFilter cameraMeshFilter;
    private MeshFilter projectorMeshFilter;

    public void Setup(cam.CameraIntrinsics cameraIntrinsics, cam.CameraIntrinsics projectorIntrinsics, int texWidth, int texHeight, Transform projectorTransform)
    {
        // Setup Camera Frustum
        if (cameraMeshFilter == null)
        {
            GameObject camObj = new GameObject("CameraFrustum");
            camObj.transform.SetParent(transform, false);
            camObj.transform.localPosition = Vector3.zero;
            camObj.transform.localRotation = Quaternion.identity;
            camObj.transform.localScale = Vector3.one;
            
            cameraMeshFilter = camObj.AddComponent<MeshFilter>();
            MeshRenderer renderer = camObj.AddComponent<MeshRenderer>();
            renderer.material = meshMaterial;
            cameraMeshFilter.mesh = CreateFrustumMesh(cameraIntrinsics, texWidth, texHeight);
        }

        // Setup Projector Frustum
        if (projectorMeshFilter == null)
        {
            GameObject projObj = new GameObject("ProjectorFrustum");
            projObj.transform.SetParent(projectorTransform, false);
            projObj.transform.localPosition = Vector3.zero;
            projObj.transform.localRotation = Quaternion.identity;
            projObj.transform.localScale = Vector3.one;

            projectorMeshFilter = projObj.AddComponent<MeshFilter>();
            MeshRenderer renderer = projObj.AddComponent<MeshRenderer>();
            renderer.material = meshMaterial;
            // Projector uses 0-255 range for intrinsics as per cam.cs
            projectorMeshFilter.mesh = CreateFrustumMesh(projectorIntrinsics, 255, 255);
        }
    }

    Mesh CreateFrustumMesh(cam.CameraIntrinsics intrinsics, float width, float height)
    {
        Mesh mesh = new Mesh();
        mesh.name = "FrustumMesh";

        Vector3 origin = Vector3.zero;
        Vector3 tl = intrinsics.PixelToCameraDirection(0, 0) * frustumLength;
        Vector3 tr = intrinsics.PixelToCameraDirection(width, 0) * frustumLength;
        Vector3 bl = intrinsics.PixelToCameraDirection(0, height) * frustumLength;
        Vector3 br = intrinsics.PixelToCameraDirection(width, height) * frustumLength;

        Vector3[] vertices = new Vector3[]
        {
            origin, tl, tr, // Top
            origin, tr, br, // Right
            origin, br, bl, // Bottom
            origin, bl, tl  // Left
        };

        int[] triangles = new int[]
        {
            0, 1, 2, // Top front
            0, 2, 1, // Top back
            3, 4, 5, // Right front
            3, 5, 4, // Right back
            6, 7, 8, // Bottom front
            6, 8, 7, // Bottom back
            9, 10, 11, // Left front
            9, 11, 10  // Left back
        };

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();

        return mesh;
    }
}
