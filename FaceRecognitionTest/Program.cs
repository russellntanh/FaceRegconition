using OpenCvSharp;
using OpenCvSharp.Face;

namespace FaceRecognitionTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            AddNewFace();

            //ExecuteFaceRecognition();
        }

        public static void ExecuteFaceRecognition()
        {
            // Initialize and train face recognizer
            var faceRecognizer = LBPHFaceRecognizer.Create();

            // Train face recognizer
            TrainFaceRecognizer(faceRecognizer);

            using (VideoCapture capture = new VideoCapture(0))
            {
                // Check camera connection
                if (!capture.IsOpened())
                {
                    Console.WriteLine("Error: could not open the webcam.");
                    return;
                }

                // Loop to continuously get frames from the webcam
                Mat originalFrame = new Mat();
                Mat grayFrame = new Mat();

                // Define scale ratio
                float scaleRatio = 1.0f; // 1.0: no resize, 2.0: double size

                // Face detection by Haar Cascade classifier
                var faceCascade = new CascadeClassifier(@"D:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml");

                while (true)
                {
                    // capture a frame
                    capture.Read(originalFrame);

                    if (originalFrame.Empty())
                    {
                        Console.WriteLine("Error: No frame data.");
                        return;
                    }

                    // Scale operation
                    Cv2.Resize(originalFrame, originalFrame, new Size(originalFrame.Width * scaleRatio, originalFrame.Height * scaleRatio));

                    // Convert to gray scale image
                    Cv2.CvtColor(originalFrame, grayFrame, ColorConversionCodes.BGR2GRAY);

                    // Gausian filter operation
                    Mat filterdFrame = new Mat();
                    Cv2.GaussianBlur(grayFrame, filterdFrame, new Size(5, 5), 0);

                    // Defect face by Haar Cascade
                    Rect[] faces = faceCascade.DetectMultiScale(grayFrame, 1.1, 4);

                    foreach (var face in faces)
                    {
                        // Crop face region only
                        using (Mat faceROI = new Mat(grayFrame, face))
                        {
                            // Recognition
                            int faceLabel = faceRecognizer.Predict(faceROI);

                            // Draw rectangle
                            Cv2.Rectangle(originalFrame, face, Scalar.Red, 2);

                            // Attach the label
                            Cv2.PutText(originalFrame, faceLabel.ToString(),
                                new Point(face.X, face.Y - 10), HersheyFonts.HersheyPlain, 1.2, Scalar.Green, 2);

                        }
                    }

                    // Show final result
                    Cv2.ImShow("Russell's webcam", originalFrame);

                    // Exit program if press ESC
                    if (Cv2.WaitKey(10) == 27)
                    {
                        break;
                    }
                }

                // Release memory
                capture.Release();
                Cv2.DestroyAllWindows();
            }
        }

        // Load training image function
        public static List<Mat> LoadTrainingFaces(string trainingFacePath)
        {
            List<Mat> trainingFaces = new List<Mat>();
            var personDir = Directory.GetDirectories(trainingFacePath);
            string folderName = Path.GetFileName(personDir[0]);

            foreach (string filePath in Directory.GetFiles(trainingFacePath +"\\"+ folderName))
            {
                Mat img = Cv2.ImRead(filePath, ImreadModes.Grayscale);
                if (img.Empty())
                {
                    Console.WriteLine($"Error: Couldn't read image {filePath}");
                    continue;
                }
                trainingFaces.Add(img);
            }

            return trainingFaces;
        }

        
        public static int[] LoadTrainingLabels()
        {
            int[] labelFaces = new int[100];

            return labelFaces;
        }
        public static void TrainFaceRecognizer(FaceRecognizer faceRecognizer)
        {
            string trainingPath = $"D:/Code/FaceRecognitionTest/FaceRecognitionTest/dataset/training";
            List<Mat> trainingFaces = LoadTrainingFaces(trainingPath);

            int[] labelFaces = LoadTrainingLabels();

            //faceRecognizer.Train(trainingFaces);
        }

        /// <summary>
        /// Press S key to save more image from the webcam
        /// </summary>
        public static void AddNewFace()
        {
            // Initialize the webcam
            VideoCapture capture = new VideoCapture(0);
            Mat frame = new Mat();
            int imageCount = 0;

            // Dictionary to store image paths and their labels
            Dictionary<string, string> imageLabelDict = new Dictionary<string, string>();

            while (true)
            {
                capture.Read(frame);
                if (!frame.Empty())
                {
                    Cv2.ImShow("Webcam", frame);
                }

                // Press 's' to save an image
                if (Cv2.WaitKey(1) == 's')
                {
                    string label = "Russell_Nguyen"; 
                    string imagePath = $"D:/images/{label}_{imageCount}.jpg";
                    imageLabelDict.Add(imagePath, label);

                    // Save the image
                    frame.SaveImage(imagePath);
                    Console.WriteLine($"Saved: {imagePath}");
                    imageCount++;
                }

                // Press 'q' to quit
                if (Cv2.WaitKey(1) == 'q')
                {
                    break;
                }
            }

            // Save the image paths and labels to a file
            SaveLabelsToFile(imageLabelDict, $"D:/images_label.txt");

            capture.Release();
            Cv2.DestroyAllWindows();
        }

        /// <summary>
        /// Save captured images and label of a person to the text file
        /// </summary>
        /// <param name="imageLabelDict"></param>
        /// <param name="v"></param>
        private static void SaveLabelsToFile(Dictionary<string, string> imageLabelDict, string txtFilePath)
        {
            using (StreamWriter sw = new StreamWriter(txtFilePath))
            {
                foreach (var entry in imageLabelDict)
                {
                    sw.WriteLine($"{entry.Key} {entry.Value}");
                }
            }
        }
    }
}
