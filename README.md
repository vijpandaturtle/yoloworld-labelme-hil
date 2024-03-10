# AI-assisted Labelling : YOLO World
A simple utility to auto-annotate images with YOLO-World &amp; export to labelme for human-in-the-loop labelling/AI-assisted labelling. 

# How to use
1. Clone the Github repository
```
git clone https://github.com/vijpandaturtle/yoloworld-labelme-hil.git
```
2. Install requirements
```
pip install -r requirements.txt
```
3. Run the script by passing in the required parameters. By default the labels will be saved in the same directory as the images.
```
python labelme-yoloworld.py --model <NAME_OF_MODEL_WEIGHT_FILE> --folder <FOLDER_PATH> --width <IMAGE_WIDTH> --height <IMAGE_HEIGHT>
```
4. Now import the directory into LabelMe, and view and edit your labels!

## References
1. [LabelMe](https://github.com/labelmeai/labelme)
