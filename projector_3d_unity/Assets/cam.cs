using UnityEngine;
using System.Collections;

public class cam : MonoBehaviour
{

    public Texture2D texture;

    public Transform projector;


    public Transform plane;

    [Header("Optimization Settings")]
    public float learningRatePos = 0.01f;
    public float learningRateRot = 1.0f;
    public float learningRateFocal = 10.0f;
    public float maxStepPos = 0.05f;
    public float maxStepRot = 2.0f;
    public float maxStepFocal = 50.0f;
    public int samplingStep = 20;
    public float rotationEps = 0.01f;
    public float focalEps = 1.0f;
    public float planeDistanceWeight = 1.0f;
    public float intersectionErrorWeight = 1.0f;

    [Header("Current Intrinsics (Display)")]
    public float camFx;
    public float camFy;

    private CameraIntrinsics cameraIntrinsics;
    private CameraIntrinsics projectorIntrinsics;

    [System.Serializable]
    public class CameraIntrinsics
    {
        public float fx;
        public float fy;
        public float cx;
        public float cy;

        public CameraIntrinsics(float fx, float fy, float cx, float cy)
        {
            this.fx = fx;
            this.fy = fy;
            this.cx = cx;
            this.cy = cy;
        }




        public Vector3 PixelToCameraDirection(float x, float y)
        {
            return new Vector3(
                (x - cx) / fx,
                -(y - cy) / fy,
                1f
            ).normalized;
        }
    }

    public class Ray
    {
        public Vector3 origin;
        public Vector3 direction;
        public void DebugDraw(Color color, float duration = 0f)
        {
            Debug.DrawRay(origin, direction * 10f, color, duration, true);
        }


        public static bool ClosestIntersection(
               Ray ray1,
               Ray ray2,
               out Vector3 point,
               out float error,
               out float t1,
               out float t2,
               Color gizmoColor,
               float duration = 0f
           )
        {
            Vector3 d1 = ray1.direction.normalized;
            Vector3 d2 = ray2.direction.normalized;

            Vector3 r = ray1.origin - ray2.origin;

            float a = Vector3.Dot(d1, d1);
            float b = Vector3.Dot(d1, d2);
            float c = Vector3.Dot(d2, d2);
            float d = Vector3.Dot(d1, r);
            float e = Vector3.Dot(d2, r);

            float denom = a * c - b * b;

            if (Mathf.Abs(denom) < 1e-6f)
            {
                point = Vector3.zero;
                error = float.PositiveInfinity;
                t1 = 0;
                t2 = 0;
                return false;
            }

            t1 = (b * e - c * d) / denom;
            t2 = (a * e - b * d) / denom;

            Vector3 p1 = ray1.origin + d1 * t1;
            Vector3 p2 = ray2.origin + d2 * t2;

            point = (p1 + p2) * 0.5f;
            error = Vector3.Distance(p1, p2);

            // 推定交点を十字で表示
            float s = 0.02f;

            Debug.DrawLine(
                point - Vector3.right * s,
                point + Vector3.right * s,
                gizmoColor,
                duration,
                true
            );

            Debug.DrawLine(
                point - Vector3.up * s,
                point + Vector3.up * s,
                gizmoColor,
                duration,
                true
            );

            Debug.DrawLine(
                point - Vector3.forward * s,
                point + Vector3.forward * s,
                gizmoColor,
                duration,
                true
            );

            return true;
        }


    }


    float ComputeCost(CameraIntrinsics cameraIntrinsics, CameraIntrinsics projectorIntrinsics)
    {
        float cost = 0f;
        int count = 0;

        for (int y = 0; y < texture.height; y++)
        {
            for (int x = 0; x < texture.width; x++)
            {
                Color color = texture.GetPixel(x, y);

                if (color.r <= 0 &&
                    color.g <= 0 &&
                    color.b <= 0)
                    continue;

                Vector3 dirCamera =
                    cameraIntrinsics.PixelToCameraDirection(x, y);

                Vector3 dirWorldCamera =
                    transform.TransformDirection(dirCamera);

                Ray rayCamera = new Ray
                {
                    origin = transform.position,
                    direction = dirWorldCamera
                };

                Vector3 dirProjector =
                    projectorIntrinsics.PixelToCameraDirection(
                        color.g * 255,
                        color.b * 255);

                Vector3 dirWorldProjector =
                    projector.transform.TransformDirection(dirProjector);

                Ray rayProjector = new Ray
                {
                    origin = projector.position,
                    direction = dirWorldProjector
                };

                if (Ray.ClosestIntersection(
                    rayCamera,
                    rayProjector,
                    out _,
                    out float error,
                    out float t1,
                    out float t2,
                    color))
                {
                    if (t1 > 0 && t2 > 0)
                    {
                        cost += error * error;
                        count++;
                    }
                }
            }
        }

        return count > 0 ? cost / count : 1000f;
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    IEnumerator Start()
    {
        cameraIntrinsics = new CameraIntrinsics(778.979f, 829.372f, texture.width / 2f, texture.height / 2f);
        projectorIntrinsics = new CameraIntrinsics(2488.889f, 2100.000f, 255f / 2, 255f / 2);

        // Add Mesh Generator
        FrustumMeshGenerator meshGen = gameObject.AddComponent<FrustumMeshGenerator>();
        // Create a basic white semi-transparent material
        Material mat = new Material(Shader.Find("Standard"));
        mat.SetFloat("_Mode", 3); // Transparent
        mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_ZWrite", 0);
        mat.DisableKeyword("_ALPHATEST_ON");
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        mat.renderQueue = 3000;
        mat.color = new Color(1, 1, 1, 0.2f);
        meshGen.meshMaterial = mat;
        meshGen.Setup(cameraIntrinsics, projectorIntrinsics, texture.width, texture.height, projector);


        while (true)
        {




            for (int y = 0; y < texture.height; y += 10)
            {
                for (int x = 0; x < texture.width; x += 10)
                {
                    Color color = texture.GetPixel(x, y);

                    // Debug.Log(
                    //     $"({x}, {y}) : " +
                    //     $"R={color.r:F3}, " +
                    //     $"G={color.g:F3}, " +
                    //     $"B={color.b:F3}, " +
                    //     $"A={color.a:F3}" +
                    // );

                    if (color.r <= 0 && color.g <= 0 && color.b <= 0) continue;


                    Vector3 dirCamera = cameraIntrinsics.PixelToCameraDirection(x, y);
                    Vector3 dirWorldCamera = transform.TransformDirection(dirCamera);

                    Ray rayCamera = new Ray
                    {
                        origin = transform.position,
                        direction = dirWorldCamera
                    };

                    float px = color.g * 255;
                    float py = color.b * 255;

                    Vector3 dirProjector = projectorIntrinsics.PixelToCameraDirection(px, py);
                    Vector3 dirWorldProjector = projector.transform.TransformDirection(dirProjector);

                    Ray rayProjector = new Ray
                    {
                        origin = projector.transform.position,
                        direction = dirWorldProjector
                    };


                    if (Ray.ClosestIntersection(rayCamera, rayProjector, out Vector3 point, out float error, out float t1, out float t2, color))
                    {
                        // Debug.Log($"Intersection Point: {point}, Error: {error}");
                    }




                }
            }
            yield return null;
        }

    }

    float ComputePlaneCost(bool drawLines = false)
    {
        if (texture == null || plane == null || cameraIntrinsics == null || projectorIntrinsics == null) return 1000f;

        float totalCost = 0f;
        int count = 0;

        Vector3 planePos = plane.position;
        Vector3 planeNormal = plane.up;

        for (int y = 0; y < texture.height; y += samplingStep)
        {
            for (int x = 0; x < texture.width; x += samplingStep)
            {
                Color color = texture.GetPixel(x, y);
                if (color.r <= 0 && color.g <= 0 && color.b <= 0) continue;

                Vector3 dirCamera = cameraIntrinsics.PixelToCameraDirection(x, y);
                Vector3 dirWorldCamera = transform.TransformDirection(dirCamera);

                Ray rayCamera = new Ray
                {
                    origin = transform.position,
                    direction = dirWorldCamera
                };

                float px = color.g * 255;
                float py = color.b * 255;

                Vector3 dirProjector = projectorIntrinsics.PixelToCameraDirection(px, py);
                Vector3 dirWorldProjector = projector.transform.TransformDirection(dirProjector);

                Ray rayProjector = new Ray
                {
                    origin = projector.transform.position,
                    direction = dirWorldProjector
                };

                if (Ray.ClosestIntersection(rayCamera, rayProjector, out Vector3 point, out float error, out float t1, out float t2, color))
                {
                    if (t1 > 0 && t2 > 0)
                    {
                        float dist = Vector3.Dot(point - planePos, planeNormal);
                        totalCost += dist * dist * planeDistanceWeight + error * error * intersectionErrorWeight;
                        count++;

                        if (drawLines)
                        {
                            Vector3 pointOnPlane = point - dist * planeNormal;
                            Debug.DrawLine(point, pointOnPlane, Color.red, 0f, true);
                        }
                    }
                }
            }
        }

        return count > 0 ? totalCost / count : 1000f;
    }

    void ApplyGradientDescent()
    {
        float eps = 0.001f;

        Vector3 originalPos = transform.position;
        Quaternion originalRot = transform.rotation;
        float originalFx = cameraIntrinsics.fx;
        float originalFy = cameraIntrinsics.fy;

        float baseCost = ComputePlaneCost(true);

        // Position gradient
        Vector3 gradPos = Vector3.zero;

        transform.position = originalPos + Vector3.right * eps;
        gradPos.x = (ComputePlaneCost() - baseCost) / eps;
        transform.position = originalPos;

        transform.position = originalPos + Vector3.up * eps;
        gradPos.y = (ComputePlaneCost() - baseCost) / eps;
        transform.position = originalPos;

        transform.position = originalPos + Vector3.forward * eps;
        gradPos.z = (ComputePlaneCost() - baseCost) / eps;
        transform.position = originalPos;

        // Rotation gradient
        Vector3 gradRot = Vector3.zero;

        transform.rotation = originalRot * Quaternion.Euler(rotationEps, 0, 0);
        gradRot.x = (ComputePlaneCost() - baseCost) / rotationEps;
        transform.rotation = originalRot;

        transform.rotation = originalRot * Quaternion.Euler(0, rotationEps, 0);
        gradRot.y = (ComputePlaneCost() - baseCost) / rotationEps;
        transform.rotation = originalRot;

        transform.rotation = originalRot * Quaternion.Euler(0, 0, rotationEps);
        gradRot.z = (ComputePlaneCost() - baseCost) / rotationEps;
        transform.rotation = originalRot;

        // Focal gradient
        Vector2 gradFocal = Vector2.zero;

        cameraIntrinsics.fx = originalFx + focalEps;
        gradFocal.x = (ComputePlaneCost() - baseCost) / focalEps;
        cameraIntrinsics.fx = originalFx;

        cameraIntrinsics.fy = originalFy + focalEps;
        gradFocal.y = (ComputePlaneCost() - baseCost) / focalEps;
        cameraIntrinsics.fy = originalFy;

        // Apply Position Update
        Vector3 moveDelta = gradPos * learningRatePos;
        if (moveDelta.magnitude > maxStepPos) moveDelta = moveDelta.normalized * maxStepPos;
        transform.position -= moveDelta;

        // Apply Rotation Update
        Vector3 rotDelta = gradRot * learningRateRot;
        if (rotDelta.magnitude > maxStepRot) rotDelta = rotDelta.normalized * maxStepRot;
        transform.rotation = originalRot * Quaternion.Euler(-rotDelta.x, -rotDelta.y, -rotDelta.z);

        // Apply Focal Update
        float stepFx = Mathf.Clamp(gradFocal.x * learningRateFocal, -maxStepFocal, maxStepFocal);
        float stepFy = Mathf.Clamp(gradFocal.y * learningRateFocal, -maxStepFocal, maxStepFocal);
        cameraIntrinsics.fx -= stepFx;
        cameraIntrinsics.fy -= stepFy;

        // Sync display variables
        camFx = cameraIntrinsics.fx;
        camFy = cameraIntrinsics.fy;

        // Debug output
        if (Time.frameCount % 60 == 0)
        {
            Debug.Log($"Cost: {baseCost:F6}, GradFocal: {gradFocal}, fx: {camFx:F2}, fy: {camFy:F2}");
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (plane != null)
        {
            ApplyGradientDescent();
        }
    }
}
