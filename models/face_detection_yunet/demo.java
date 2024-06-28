import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.UnixStyleUsageFormatter;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_SIMPLEX;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public class demo {

    // Valid combinations of backends and targets
    static int[][] backendTargetPairs = {
            {opencv_dnn.DNN_BACKEND_OPENCV, opencv_dnn.DNN_TARGET_CPU},
            {opencv_dnn.DNN_BACKEND_CUDA, opencv_dnn.DNN_TARGET_CUDA},
            {opencv_dnn.DNN_BACKEND_CUDA, opencv_dnn.DNN_TARGET_CUDA_FP16},
            {opencv_dnn.DNN_BACKEND_TIMVX, opencv_dnn.DNN_TARGET_NPU},
            {opencv_dnn.DNN_BACKEND_CANN, opencv_dnn.DNN_TARGET_NPU}
    };

    static class Args {
        @Parameter(names = {"--help", "-h"}, order = 0, help = true,
                description = "Print help message.")
        boolean help;
        @Parameter(names = {"--input", "-i"}, order = 1,
                description = "Set input to a certain image, omit if using camera.")
        String input;
        @Parameter(names = {"--model", "-m"}, order = 2,
                description = "Set model type.")
        String model = "face_detection_yunet_2023mar.onnx";
        @Parameter(names = {"--backend_target", "-bt"}, order = 3,
                description = "Choose one of the backend-target pair to run this demo:" +
                        " 0: OpenCV implementation + CPU," +
                        " 1: CUDA + GPU (CUDA), " +
                        " 2: CUDA + GPU (CUDA FP16)," +
                        " 3: TIM-VX + NPU," +
                        " 4: CANN + NPU")
        int backendTarget = 0;
        @Parameter(names = {"--conf_threshold"}, order = 5,
                description = "Set the minimum needed confidence for the model to identify a face. Filter out faces of conf < conf_threshold")
        float confThreshold = 0.9f;
        @Parameter(names = {"--nms_threshold"}, order = 5,
                description = "Set the threshold to suppress overlapped boxes. Suppress boxes if IoU(box1, box2) >= nms_threshold, the one of higher score is kept.")
        float nmsThreshold = 0.3f;
        @Parameter(names = {"--top_k"}, order = 5,
                description = "Keep top_k bounding boxes before NMS. Set a lower value may help speed up postprocessing.")
        int topK = 5000;
        @Parameter(names = {"--save", "-s"}, order = 4,
                description = "Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.")
        boolean save;
        @Parameter(names = {"--vis", "-v"}, order = 5, arity = 1,
                description = "Specify to open a new window to show results. Invalid in case of camera input.")
        boolean vis = true;
    }


    static class YuNet {
        private final FaceDetectorYN model;

        YuNet(String modelPath, Size inputSize, float confThreshold, float nmsThreshold, int topK,
              int backendId, int targetId) {
            model = FaceDetectorYN.create(modelPath, "", inputSize, confThreshold, nmsThreshold, topK,
                    backendId, targetId);
        }

        void setInputSize(Size inputSize) {
            model.setInputSize(inputSize);
        }

        Mat infer(Mat image) {
            final Mat res = new Mat();
            model.detect(image, res);
            return res;
        }
    }

    static Mat visualize(Mat image, Mat faces, Scalar textColor, double fps) {
        final Mat output = image.clone();
        final int h = output.rows();
        final int w = output.cols();
        if (fps >= 0) {
            putText(output, String.format("FPS: %.2f", fps), new Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, textColor);
        }

        // TODO
        return output;
    }

    public static void main(String[] argv) {
        final Args args = new Args();
        final JCommander jc = JCommander.newBuilder()
                .addObject(args)
                .build();
        jc.setUsageFormatter(new UnixStyleUsageFormatter(jc));
        jc.parse(argv);
        if (args.help) {
            jc.usage();
            return;
        }
        final int backendId = backendTargetPairs[args.backendTarget][0];
        final int targetId = backendTargetPairs[args.backendTarget][1];
        final YuNet model = new YuNet(args.model, new Size(320, 320), args.confThreshold, args.nmsThreshold,
                args.topK, backendId, targetId);

        if (args.input != null) {
            final Mat image = imread(args.input);

            // Inference
            model.setInputSize(image.size());
            final Mat faces = model.infer(image);

            // Print faces
            System.out.printf("%d faces detected:\n", faces.rows());
            for (int i = 0; i < faces.rows(); i++) {
                // TODO
            }

            // Draw reults on the input image
            if (args.save || args.vis) {
                final Mat resImage = visualize(image, faces, new Scalar(0, 0, 255, 0), -1);
                if (args.save) {
                    System.out.println("Results are saved to result.jpg");
                    imwrite("result.jpg", resImage);
                }
                if (args.vis) {
                    namedWindow(args.input, WINDOW_AUTOSIZE);
                    imshow(args.input, resImage);
                    waitKey(0);
                }
            }

        } else { // // Call default camera
            final int deviceId = 0;
            final VideoCapture cap = new VideoCapture(deviceId);
            final int w = (int) cap.get(CAP_PROP_FRAME_WIDTH);
            final int h = (int) cap.get(CAP_PROP_FRAME_HEIGHT);
            model.setInputSize(new Size(w, h));

            final TickMeter tm = new TickMeter();
            final Mat frame = new Mat();
            while (waitKey(1) < 0) {
                boolean hasFrame = cap.read(frame);
                if (!hasFrame) {
                    System.out.println("No frames grabbed! Exiting ...");
                    break;
                }
                // Inference
                tm.start();
                final Mat faces = model.infer(frame);
                tm.stop();
                final Mat resImage = visualize(frame, faces, new Scalar(0, 0, 255, 0), tm.getFPS());
                imshow("YuNet Demo", resImage);

                tm.reset();
            }
        }
    }


}
