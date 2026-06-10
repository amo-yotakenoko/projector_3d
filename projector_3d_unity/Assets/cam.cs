using UnityEngine;
using System.Collections;

public class cam : MonoBehaviour
{

    public Texture2D texture;

    public Transform projector;

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
                return false;
            }

            float t1 = (b * e - c * d) / denom;
            float t2 = (a * e - b * d) / denom;

            Vector3 p1 = ray1.origin + d1 * t1;
            Vector3 p2 = ray2.origin + d2 * t2;

            point = (p1 + p2) * 0.5f;
            error = Vector3.Distance(p1, p2);

            // 誤差部分を赤で表示
            // Debug.DrawLine(
            //     p1,
            //     p2,
            //     Color.red,
            //     duration,
            //     true
            // );

            // 推定交点を十字で表示
            float s = 0.02f;

            Debug.DrawLine(
                point - Vector3.right * s,
                point + Vector3.right * s,
                Color.white,
                duration,
                true
            );

            Debug.DrawLine(
                point - Vector3.up * s,
                point + Vector3.up * s,
                Color.white,
                duration,
                true
            );

            Debug.DrawLine(
                point - Vector3.forward * s,
                point + Vector3.forward * s,
                Color.white,
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
                    out float error))
                {
                    cost += error * error;
                    count++;
                }
            }
        }

        return count > 0 ? cost / count : float.MaxValue;
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    IEnumerator Start()
    {
        CameraIntrinsics cameraIntrinsics = new CameraIntrinsics(778.979f, 829.372f, texture.width / 2f, texture.height / 2f);
        CameraIntrinsics projectorIntrinsics = new CameraIntrinsics(2488.889f, 2100.000f, 255f / 2, 255f / 2);


        while (true)
        {




            for (int y = 0; y < texture.height; y += 5)
            {
                for (int x = 0; x < texture.width; x += 5)
                {
                    Color color = texture.GetPixel(x, y);

                    // Debug.Log(
                    //     $"({x}, {y}) : " +
                    //     $"R={color.r:F3}, " +
                    //     $"G={color.g:F3}, " +
                    //     $"B={color.b:F3}, " +
                    //     $"A={color.a:F3}"
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

                    if (x == 0 || x + 5 >= texture.width || y == 0 || y + 5 >= texture.height)
                        rayCamera.DebugDraw(color);

                    if (px <= 1.0f || px >= 254.0f || py <= 1.0f || py >= 254.0f)
                        rayProjector.DebugDraw(color);


                    if (Ray.ClosestIntersection(rayCamera, rayProjector, out Vector3 point, out float error))
                    {
                        // Debug.Log($"Intersection Point: {point}, Error: {error}");
                    }




                }
            }
            yield return null;
        }

    }

    // Update is called once per frame
    void Update()
    {

    }
}
