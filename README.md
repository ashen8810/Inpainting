# Inpainting

Remove signatures and unnecessary parts from images.

## Description

This project provides a simple and effective way to remove small watermarks, signatures, or other unwanted elements from images. By leveraging image processing techniques with OpenCV, users can manually select the area where the watermark or unwanted element is located, and the algorithm will inpaint the selected region to seamlessly blend it with the surrounding pixels.

## Features

- **User-Friendly Interface:** Easily select the area containing the watermark or unwanted element.
- **Efficient Inpainting:** Utilizes advanced image processing techniques in OpenCV to fill in the selected area.
- **Quality Results:** Produces images with minimal traces of the removed elements, maintaining the overall quality of the original image.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/inpainting.git](https://github.com/ashen8810/Inpainting.git
   ```
2. Navigate to the project directory:
   ```bash
    cd Inpainting/Project Final
   ```
3. Install the required packages:
  ```bash
    pip install -r requirements.txt
  ```

## Usage
1. Change the image path in the code
2. Run the python file
```python
  python inpainting.py
```
3. Paint on the watermark

![Before](https://github.com/ashen8810/Inpainting/blob/main/Before.png)

4. Press escape button

![Results](https://github.com/ashen8810/Inpainting/blob/main/Results.png)



## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
