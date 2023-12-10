# SNU CSE Computer Vision Project
# ImmerVision - Real Time 3D HoloProjection

**Ji Hun Seo (서지훈, 2021-19397)** markseo0424@snu.ac.kr \
**Moon Seok Khang (강문석, 2022-15470)** mskhang@snu.ac.kr \
**Na Rim Kim (김나림, 2022-17320)** wingedlz@snu.ac.kr \
**Woo Hyun Kim (김우현, 2022-13625)** tryyoung@snu.ac.kr

## Summary

Our focus is to implement a virtual 3D holographic environment without the aid of virtual reality devices.
By tracking the eye movement of the user in real-time, this program will output images that would trick the user into feeling that an object actually exist beyond the screen.
Furthermore, we added a lighting detection feature so that the lighting environment surrounding the user gets reflected on to the virtual images inside the screen.
Overall, these features will give the user the impression that the object beyond the screen is sharing the same space with the user.

## Key Concepts

ImmerVision uses concepts that were covered in the Computer Vision course at SNU CSE.
The key concepts used are:
- Lambertian (Relections and Colors)
- Projection
- Camera Coordinates and World Coordinates


## Libraries Used
Use the following command on the command line to install the libraries.
(For PyTorch, there is a separate command line.)
```commandline
pip install -r requirements.txt
```

**opencv** - For accessing the webcam, simple image processing, image output

- Reference - https://076923.github.io/posts/Python-opencv-2/

**numpy** - Utilized for matrix operations (matrix multiplication, etc.)

**mediapipe** - Library for detecting images and eyes.

**torch** - Despite being widely used for deep-learning, for our project this library will be
used to perform fast matrix operations using the GPU of the PC.
```commandline
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 ```

## Project File Structure

### Detection
1. Uses the webcam to detect the eye position
2. Calculates the horizontal and vertical position of the user's eye

- `built_in/face_detection_mediapipe.py`: Detects the eye position and output the eye position in camera pixels
- `render/environment.py`: Handles the lighting, and Lambertian calculations

### Rendering
3. 3D Projection
4. Rendering the lighting

- `render/light_ray_render.py`: Calculates the projection of the virtual object on the screen, finds the depth of the polygons and renders them

### Others
- `cv_functions/capture_video.py`: Gets the eye position information from `face_detection_mediapipe.py` and sends the eye position data to `light_ray_render.py`. This is the top of the program, where all the iteration jobs happen.
- `render/open_obj.py`: Reads the .obj file.
There are some constraints with the .obj file for ImmerVision. All the vertices must be defined before the vertex normals and the surfaces. Also, all the vertex normals must be defined before the surfaces/


## How to Run

Run `main.py` to start the program.
