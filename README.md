# 🎨 Background Replacement Tool

An **interactive image processing application** built with Streamlit that allows you to automatically replace backgrounds in images using **HSV color space segmentation** and **morphological operations**. Perfect for creating professional photos, removing unwanted backgrounds, and creative image editing.

---

## ✨ Features

- 🖼️ **Dual Image Upload**: Upload foreground and background images
- 🎯 **HSV Color Detection**: Precise background color targeting
- 🔧 **Real-time Parameter Tuning**: Interactive sliders for fine control
- 🧹 **Morphological Operations**: Noise reduction and mask refinement
- 📱 **Live Preview**: See results instantly as you adjust parameters
- 💾 **Download Results**: Export final images as PNG
- 🎨 **Visual Pipeline**: Step-by-step processing visualization

---

## 🛠️ Tech Stack

- **Streamlit** - Interactive web application
- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computations
- **PIL (Pillow)** - Image handling and export
- **Matplotlib** - Visualization support

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation
```bash
git clone https://github.com/RahulKumar2340029/Image-Segmentation.git
cd Image-Segmentation
pip install streamlit opencv-python numpy pillow matplotlib
```

### Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 How to Use

### Step 1: Upload Images
- **Foreground Image**: Upload the image with the background you want to remove
- **Background Image**: Upload the new background you want to add

### Step 2: Adjust HSV Parameters
Use the sidebar sliders to define the background color range:
- **Hue**: Color type (0-179)
- **Saturation**: Color intensity (0-255) 
- **Value**: Brightness (0-255)

### Step 3: Fine-tune Morphology
- **Enable/Disable**: Toggle morphological operations
- **Closing Kernel**: Fill gaps in background detection
- **Opening Kernel**: Remove noise and small artifacts

### Step 4: Download Result
Click "Download Result" to save your processed image

---

## 🧠 How It Works

### Image Processing Pipeline:
1. **Preprocessing**: Apply Gaussian blur to reduce noise
2. **Color Space Conversion**: Convert BGR to HSV for better color separation
3. **Background Masking**: Create mask using HSV color thresholding
4. **Morphological Refinement**: Apply closing and opening operations
5. **Foreground Extraction**: Invert mask to get foreground
6. **Background Replacement**: Combine foreground with new background

### Mathematical Operations:
- **HSV Thresholding**: `mask = cv2.inRange(hsv, lower_bound, upper_bound)`
- **Morphological Closing**: Fills small gaps in detected regions
- **Morphological Opening**: Removes small noise elements
- **Bitwise Operations**: Combine foreground and background seamlessly

---

## 🎯 Best Practices

### For Best Results:
- **Uniform Backgrounds**: Solid colors work better than complex patterns
- **Good Lighting**: Even lighting reduces shadows and color variations
- **Contrasting Colors**: Foreground should contrast well with background
- **High Resolution**: Better quality images produce cleaner results

### Parameter Tuning Tips:
- Start with **wide HSV ranges** and narrow down gradually
- Use **larger closing kernels** for backgrounds with gaps
- Use **smaller opening kernels** to preserve detail
- **Preview masks** to verify detection accuracy

---

## 🎨 Use Cases

### Professional Photography:
- Product photography background removal
- Portrait background replacement
- Real estate photo enhancement

### Creative Projects:
- Artistic photo compositions
- Social media content creation
- Digital art and design

### Educational Applications:
- Computer vision learning
- Image processing demonstrations
- HSV color space understanding

---

## 📊 Supported Formats

### Input Formats:
- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **High Resolution**: Up to 4K images supported

### Output Format:
- **PNG** with transparency support
- **Lossless compression** for best quality

---

## ⚙️ Technical Details

### HSV Color Space Advantages:
- **Intuitive**: Separates color information from lighting
- **Robust**: Less sensitive to lighting variations
- **Effective**: Better for color-based segmentation than RGB

### Morphological Operations:
- **Closing**: Connects nearby regions, fills small holes
- **Opening**: Removes small noise, separates connected objects
- **Kernel Size**: Larger kernels = stronger effects

---

## 📈 Performance Notes

- **Processing Speed**: Real-time for images up to 2MP
- **Memory Usage**: Scales with image resolution
- **Browser Compatibility**: Works with all modern browsers
- **Mobile Support**: Responsive design for mobile devices

---

## 🚀 Future Enhancements

### Planned Features:
- [ ] **Multiple Background Detection**: Handle complex multi-colored backgrounds
- [ ] **Edge Refinement**: Smoother foreground edges using blur/feathering
- [ ] **Batch Processing**: Process multiple images simultaneously
- [ ] **Advanced Filters**: Gaussian blur, sharpen, color correction
- [ ] **Preset Templates**: Common background removal scenarios

### Technical Improvements:
- [ ] **GPU Acceleration**: Faster processing for large images
- [ ] **Machine Learning**: AI-based background detection
- [ ] **Video Support**: Background replacement for video files
- [ ] **API Integration**: RESTful API for developers

---

## 🐛 Troubleshooting

### Common Issues:

**Background not detected properly:**
- Adjust HSV ranges using the sliders
- Try different lighting conditions
- Use more uniform background colors

**Edges look rough:**
- Increase closing kernel size
- Apply additional blur preprocessing
- Use higher resolution images

**Foreground parts missing:**
- Narrow down HSV ranges
- Disable morphological operations temporarily
- Check for color similarity between foreground and background

**App running slowly:**
- Reduce image resolution before upload
- Close other browser tabs
- Check system memory usage

---

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your improvements
4. **Test** thoroughly with various images
5. **Submit** a pull request

### Areas for Contribution:
- Algorithm improvements
- UI/UX enhancements
- Performance optimizations
- Documentation updates
- Bug fixes and testing

---

## 📜 License

MIT License © 2024 Rahul Kumar

---

## 🙏 Acknowledgments

- **OpenCV Community** for excellent computer vision tools
- **Streamlit Team** for the amazing web framework
- **Contributors** who help improve this tool

---

<div align="center">

**⭐ Star this repo if you find it useful!**

*Perfect for photographers, designers, and anyone who needs quick background replacement*

[View Repository](https://github.com/RahulKumar2340029/Image-Segmentation) • [Report Bug](https://github.com/RahulKumar2340029/Image-Segmentation/issues) • [Request Feature](https://github.com/RahulKumar2340029/Image-Segmentation/issues)

</div>
