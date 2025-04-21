We're happy to share that the first test version of our face recognition system is done! This early version still has some rough spots, but it already does something cool: you can upload a photo of someone, and the system will find and show both the face and the text in the picture.

We call this project the Facial Identity Recognition System. It's just the beginning. In the future, we plan to make it even smarter. For example, the system will be able to tell if a real person is standing in front of the camera (not just a photo), and it will check if the person's selfie matches the photo on their ID card.

We’re also working on making it faster. We want it to process images more quickly and load the models faster so it can work better when many people use it at the same time.

To read the text on ID cards, we use two OCR (optical character recognition) tools: EasyOCR and HezarOCR. These are great at reading Persian writing. For finding faces and ID card parts, we use a special model based on YOLO that we trained ourselves, called "polov." If you're curious, you can find this model in our GitHub project.

The backend of the app runs on Flask, a simple but powerful Python tool for building web apps. It takes care of uploading files, managing tasks, and handling things in the background. YOLO also helps the web part of the app know where the faces and text are.

This version is just a test, so it’s not ready for real-world use yet. But it gives us a strong base to keep building on. We’d love your help! Try it out, give us your thoughts, and maybe even contribute. You can find everything, including the code and instructions, at: github.com/Nobiqe/atuomate-clean.

Thanks for checking it out—we’re excited for what comes next!

