# Application Setup and Execution

## Steps to Run the Application

1. **Download the Dataset**  
   - Obtain the Flickr8k dataset and store the images in a folder named `Flicker8k_Dataset` within the project directory.  
   - Each image should have the path structure: `Flicker8k_Dataset/<image_name>.jpg`.

2. **Generate Output and Checkpoints**  
   - Open and execute each cell in the provided `.ipynb` notebook. This step allows you to view the output generated at each stage and to create checkpoints during model training.  
   - The application will automatically save the 5 most recent checkpoints in a newly created `checkpoints` directory.  
   - **Note:** If you only wish to run the application without additional training, pre-generated captions are stored in `data.json`.

3. **Run the Server**  
   - Start the application by running `server.py`, which contains the backend code.

## Application Functionality

Upon uploading images, the application generates captions and saves them along with the image name as a JSON object. Uploaded images are also stored in a `gallery` folder for future reference.