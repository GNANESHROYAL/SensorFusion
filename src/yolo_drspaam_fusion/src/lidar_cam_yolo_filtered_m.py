import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf2_ros
import tf2_geometry_msgs  # For point transformation
import geometry_msgs.msg
from ultralytics import YOLO  # Import YOLOv8
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from message_filters import ApproximateTimeSynchronizer, Subscriber

class ScanImageOverlayWithYOLO(Node):
    def __init__(self):
        super().__init__('scan_image_overlay_yolo_node')

        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Ensure 'yolov8n.pt' is in the correct path

        # Subscribers for LaserScan and Camera Image using message_filters for synchronization
        self.scan_sub = Subscriber(self, LaserScan, '/scan')
        self.image_sub = Subscriber(self, Image, '/camera/image_raw')

        self.ts = ApproximateTimeSynchronizer(
            [self.scan_sub, self.image_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        self.bridge = CvBridge()
        self.latest_image = None  # Store the latest camera image

        # TF2 Buffer and Listener for frame transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers for visualization markers and point clouds
        self.marker_pub = self.create_publisher(MarkerArray, '/human_markers', 10)
        self.all_pointcloud_pub = self.create_publisher(PointCloud2, '/transformed_lidar_points', 10)
        self.filtered_pointcloud_pub = self.create_publisher(PointCloud2, '/filtered_lidar_points', 10)

        # Camera Intrinsic Parameters (calculated from URDF)
        self.fx = 533.33  # Focal length in pixels along X-axis
        self.fy = 533.33  # Focal length in pixels along Y-axis
        self.cx = 320.0    # Principal point X-coordinate in pixels
        self.cy = 240.0    # Principal point Y-coordinate in pixels

        # Variables to store transformed LiDAR points
        self.transformed_points = None
        self.valid_3d_points = None

    def synced_callback(self, scan_msg, image_msg):
        # Convert ROS2 Image message to OpenCV image
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            self.get_logger().info("Received synchronized scan and image messages.")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Process LaserScan data
        try:
            # Wait for the transform from 'lidar_link' to 'camera_link'
            transform = self.tf_buffer.lookup_transform(
                'camera_link',  # Target frame
                scan_msg.header.frame_id,  # Source frame (e.g., 'lidar_link')
                rclpy.time.Time(),  # Latest available transform
                timeout=rclpy.duration.Duration(seconds=1.0)  # Timeout handling
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.TimeoutException) as e:
            self.get_logger().warn(f"Transform lookup failed: {e}")
            return

        # Convert LaserScan data to X-Y-Z coordinates
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        ranges = np.array(scan_msg.ranges)
        valid_indices = np.isfinite(ranges) & (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)

        ranges = ranges[valid_indices]
        angles = angles[valid_indices]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)  # Assuming 2D LiDAR scanning in X-Y plane

        # Apply transformation to align LiDAR data with the camera frame
        transformed_points = []
        for xi, yi, zi in zip(x, y, z):
            point = geometry_msgs.msg.PointStamped()
            point.header.frame_id = scan_msg.header.frame_id
            point.point.x = xi
            point.point.y = yi
            point.point.z = zi

            # Use tf2_geometry_msgs to transform the point
            try:
                transformed_point = tf2_geometry_msgs.do_transform_point(point, transform)
                transformed_points.append([
                    transformed_point.point.x,
                    transformed_point.point.y,
                    transformed_point.point.z
                ])
            except Exception as e:
                self.get_logger().warn(f"Point transformation failed: {e}")
                continue

        self.transformed_points = np.array(transformed_points)
        self.get_logger().info(f"Number of transformed LiDAR points: {len(self.transformed_points)}")
        if len(self.transformed_points) > 0:
            self.get_logger().info(f"Sample transformed point: {self.transformed_points[0]}")

        # Publish all transformed LiDAR points for visualization
        self.publish_all_transformed_points(self.transformed_points)

        # Proceed to overlay
        self.overlay_lidar_and_yolo_on_image()

    def overlay_lidar_and_yolo_on_image(self):
        """Overlay filtered LiDAR points and YOLO bounding boxes on the camera image."""
        if self.transformed_points is None or self.latest_image is None:
            self.get_logger().warn("Insufficient data for overlay.")
            return

        # Perform YOLO detection
        image_rgb = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb)

        boxes = []
        # Draw YOLO bounding boxes on the image
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])

                if class_id == 0:  # Class 0 is 'person' in YOLO
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image_rgb,
                        f'Person: {confidence:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    boxes.append(box)

        if not boxes:
            self.get_logger().warn("No 'person' detections from YOLO.")
            # Optionally, delete all existing markers if no detections
            self.delete_all_markers()
            return

        # Project LiDAR points to image
        image_points = self.project_lidar_to_image(self.transformed_points)

        if image_points.size == 0:
            self.get_logger().warn("No image points available for overlay.")
            return

        # Filter LiDAR points to show only those inside YOLO bounding boxes
        filtered_points = self.filter_lidar_points_in_bbox(image_points, boxes, self.valid_3d_points)

        # Publish markers for the filtered LiDAR points
        self.publish_human_markers(filtered_points)

        # Optionally, visualize the overlay
        self.visualize_overlay(image_rgb, image_points, boxes, filtered_points)

    def project_lidar_to_image(self, points_3d):
        """Project LiDAR points onto the image plane using camera calibration parameters."""
        if points_3d.size == 0:
            self.get_logger().warn("No points available for projection.")
            return np.array([])

        # Camera intrinsic parameters
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy

        self.get_logger().info(f"Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

        # Convert ROS camera_link frame to computer vision frame
        # Mapping:
        # x_cv = -Y_ros (left to right)
        # y_cv = -Z_ros (up to down)
        # z_cv = X_ros (forward)

        x_cv = -points_3d[:, 1]  # Negative Y_ros
        y_cv = -points_3d[:, 2]  # Negative Z_ros
        z_cv = points_3d[:, 0]   # X_ros

        # Avoid division by zero and exclude points behind the camera
        valid_depth = z_cv > 0.1  # Assuming camera can't see points closer than 0.1m
        x_cv = x_cv[valid_depth]
        y_cv = y_cv[valid_depth]
        z_cv = z_cv[valid_depth]

        if len(z_cv) == 0:
            self.get_logger().warn("All LiDAR points are behind the camera.")
            return np.array([])

        # Project onto image plane
        u = (fx * x_cv / z_cv) + cx
        v = (fy * y_cv / z_cv) + cy

        image_points = np.column_stack((u, v))

        # Filter points that are within image bounds
        image_shape = self.latest_image.shape
        valid_indices = (
            (u >= 0) & (u < image_shape[1]) &
            (v >= 0) & (v < image_shape[0])
        )

        image_points = image_points[valid_indices]
        self.valid_3d_points = points_3d[valid_depth][valid_indices]

        self.get_logger().info(f"Number of projected image points: {len(image_points)}")
        if len(image_points) > 0:
            self.get_logger().info(f"Sample projected point: {image_points[0]}")

        return image_points.astype(int)

    def filter_lidar_points_in_bbox(self, image_points, boxes, original_points):
        """Filter LiDAR points to keep only those inside the YOLO bounding boxes."""
        filtered_points_list = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            self.get_logger().info(f"Processing bounding box: ({x1}, {y1}) to ({x2}, {y2})")
            # Filter LiDAR points inside the bounding box
            mask = (
                (image_points[:, 0] >= x1) &
                (image_points[:, 0] <= x2) &
                (image_points[:, 1] >= y1) &
                (image_points[:, 1] <= y2)
            )
            filtered_points = original_points[mask]
            self.get_logger().info(f"Points inside bounding box: {len(filtered_points)}")
            filtered_points_list.append(filtered_points)

        return filtered_points_list

    def publish_human_markers(self, filtered_points_list):
        """Publish markers at the positions of filtered LiDAR points with an offset."""
        marker_array = MarkerArray()

        if len(filtered_points_list) == 0:
            # No detections; delete all markers
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            self.marker_pub.publish(marker_array)
            self.get_logger().info("No detections. Deleted all markers.")
            return

        # Define the offsets
        x_offset = -1.0  # Adjust this value as needed
        y_offset = 0  # Adjust this value as needed

        # Fixed marker ID for single person
        marker_id = 0

        for filtered_points in filtered_points_list:
            if len(filtered_points) == 0:
                continue

            # Compute the centroid of the points in 'camera_link' frame
            centroid = np.mean(filtered_points, axis=0)
            self.get_logger().info(f"Centroid position in camera_link frame: x={centroid[0]:.2f}, y={centroid[1]:.2f}, z={centroid[2]:.2f}")

            # Create a PointStamped for the centroid
            centroid_point = geometry_msgs.msg.PointStamped()
            centroid_point.header.frame_id = 'camera_link'
            centroid_point.header.stamp = self.get_clock().now().to_msg()
            centroid_point.point.x = centroid[0]
            centroid_point.point.y = centroid[1]
            centroid_point.point.z = centroid[2]

            # Transform centroid to 'odom' frame
            try:
                transformed_centroid = self.tf_buffer.transform(
                    centroid_point,
                    'odom',  # Replace with 'map' or 'world' if appropriate
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                self.get_logger().info(f"Centroid position in odom frame before offset: x={transformed_centroid.point.x:.2f}, y={transformed_centroid.point.y:.2f}, z={transformed_centroid.point.z:.2f}")
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.TimeoutException) as e:
                self.get_logger().warn(f"Centroid transform failed: {e}")
                continue

            # Apply the offset
            transformed_centroid.point.x += x_offset
            transformed_centroid.point.y += y_offset
            self.get_logger().info(f"Centroid position in odom frame after offset: x={transformed_centroid.point.x:.2f}, y={transformed_centroid.point.y:.2f}, z={transformed_centroid.point.z:.2f}")

            # Create a marker in the 'odom' frame
            marker = Marker()
            marker.header.frame_id = 'odom'  # Use the frame you transformed to
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'human_markers'
            marker.id = marker_id  # Fixed ID
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = transformed_centroid.point.x
            marker.pose.position.y = transformed_centroid.point.y
            marker.pose.position.z = transformed_centroid.point.z
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3  # Adjust size as needed
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0  # Alpha
            marker.color.r = 0.0  # Red
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)
            marker_id += 1  # Increment for multiple detections

        if len(marker_array.markers) > 0:
            self.marker_pub.publish(marker_array)
            self.get_logger().info(f"Published {len(marker_array.markers)} human markers.")

            # Optionally, publish PointCloud2 for visualization
            all_filtered_points = np.vstack([fp for fp in filtered_points_list if len(fp) > 0])
            self.publish_filtered_pointcloud(all_filtered_points)



    def publish_all_transformed_points(self, points):
        """Publish all transformed LiDAR points for visualization."""
        if points.size == 0:
            self.get_logger().warn("No transformed LiDAR points to publish.")
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'  # Ensure correct frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pointcloud = pc2.create_cloud(header, fields, points.tolist())
        self.all_pointcloud_pub.publish(pointcloud)
        self.get_logger().info(f"Published {len(points)} transformed LiDAR points to /transformed_lidar_points.")

    def publish_filtered_pointcloud(self, points):
        """Publish PointCloud2 message for filtered LiDAR points."""
        if points.size == 0:
            self.get_logger().warn("No filtered LiDAR points to publish.")
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'  # Ensure correct frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pointcloud = pc2.create_cloud(header, fields, points.tolist())
        self.filtered_pointcloud_pub.publish(pointcloud)
        self.get_logger().info(f"Published {len(points)} points to /filtered_lidar_points.")

    def delete_all_markers(self):
        """Delete all existing markers."""
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array)
        self.get_logger().info("Deleted all markers.")

    def visualize_overlay(self, image_rgb, image_points, boxes, filtered_points_list):
        """Optionally visualize the overlay of LiDAR points and detections."""
        plt.figure(figsize=(8,6))
        plt.imshow(image_rgb)

        # Plot all projected LiDAR points
        plt.scatter(image_points[:, 0], image_points[:, 1], color='blue', s=2, label='LiDAR Points')

        # Plot filtered points in red
        for filtered_points in filtered_points_list:
            if len(filtered_points) > 0:
                # Re-project to image plane for visualization
                projected_points = self.project_lidar_to_image(filtered_points)
                if projected_points.size > 0:
                    plt.scatter(projected_points[:, 0], projected_points[:, 1], color='red', s=5, label='Filtered Points')

        plt.title("YOLO Detections and Filtered LiDAR Points")
        plt.axis('off')  # Hide axes
        plt.legend(loc='upper right')
        plt.savefig('lidar_camera_overlay.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        self.get_logger().info('Overlay image saved as lidar_camera_overlay.png')

def main(args=None):
    rclpy.init(args=args)
    node = ScanImageOverlayWithYOLO()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
