# Computer Vision Accuracy Report

**Generated:** 2025-09-17 08:01:55

## Model Information
- **Model Path:** ../../yolov8n.pt
- **Test Images:** test_images
- **Total Images:** 30

## Performance Metrics
- **Total Vehicles Detected:** 317
- **Average Vehicles per Image:** 10.57
- **Average Processing Time:** 0.516s
- **FPS (Frames Per Second):** 1.94
- **Min Processing Time:** 0.438s
- **Max Processing Time:** 1.158s

## Performance Analysis
❌ **FPS Performance:** Needs improvement (<10 FPS)
- **Processing Speed:** 1.94 FPS
- **Detection Rate:** 10.57 vehicles per image

## Detailed Results
| Image | Vehicles | Processing Time (s) | Image Size |
|-------|----------|-------------------|------------|
| adhitya-ginanjar-FU6LSA6TFg8-unsplash.jpg | 3 | 1.158 | 2748x2061 |
| al-khaff-q8X-rzYht4o-unsplash.jpg | 10 | 0.552 | 4946x3297 |
| aleksei-zaitcev-OpsqETHSsVw-unsplash.jpg | 12 | 0.572 | 3056x4592 |
| alex-gurung-CuMN4xYb-e8-unsplash (1).jpg | 14 | 0.556 | 3919x3135 |
| alex-gurung-CuMN4xYb-e8-unsplash.jpg | 14 | 0.497 | 3919x3135 |
| aniket-singh-VUNoTFpncKk-unsplash.jpg | 11 | 0.512 | 4032x3024 |
| ankit-shetty-_c3S8_-XjGI-unsplash.jpg | 23 | 0.516 | 3264x4896 |
| arto-suraj-tkTSWcoXluQ-unsplash.jpg | 8 | 0.468 | 3995x5993 |
| atharva-tulsi-poiZagysHEY-unsplash.jpg | 18 | 0.466 | 4189x3120 |
| bornil-amin-GNcDnsQEaE8-unsplash.jpg | 13 | 0.504 | 2916x5184 |
| nomadic-julien-GTMjxwhRt-Q-unsplash.jpg | 13 | 0.438 | 6000x4000 |
| people-driving-cars-city-street.jpg | 15 | 0.458 | 4024x6048 |
| pexels-stijn-dijkstra-1306815-15528021.jpg | 5 | 0.464 | 6766x4511 |
| piero-regnante-Rg_29ECKqzA-unsplash (1).jpg | 10 | 0.482 | 2592x3872 |
| piero-regnante-Rg_29ECKqzA-unsplash.jpg | 10 | 0.453 | 2592x3872 |
| raghav-triapthi-lcKQLzvod5A-unsplash.jpg | 15 | 0.545 | 4032x3024 |
| ravi-sharma-XBN7w2vuiO8-unsplash.jpg | 0 | 0.459 | 5184x3456 |
| refhad-GQMOID7qa2o-unsplash.jpg | 2 | 0.467 | 5148x3432 |
| rohit-durbha-Vn69o53l908-unsplash.jpg | 29 | 0.524 | 4000x3000 |
| sagar-bhujel-1c1c9afYx9w-unsplash.jpg | 27 | 0.522 | 4000x3000 |
| shohidul-alam-vSwJHOC7EA8-unsplash.jpg | 2 | 0.498 | 4032x3024 |
| shubham-dhage-KWj4NPwSQkc-unsplash.jpg | 0 | 0.524 | 6831x5152 |
| simon-reza-bbAduDc5RuM-unsplash.jpg | 5 | 0.516 | 5938x3959 |
| soham-banerjee-d6GF6DW7QA8-unsplash.jpg | 0 | 0.462 | 3386x5092 |
| tony-sebastian-DYfT2tK6Rrw-unsplash.jpg | 6 | 0.452 | 5600x4000 |
| urban-landscape-japan-cars.jpg | 21 | 0.455 | 4045x2890 |
| vu-le-fNMpevndtDk-unsplash.jpg | 8 | 0.452 | 4032x3024 |
| yash-rawat-nIPV1q-TfiM-unsplash.jpg | 4 | 0.516 | 4032x2268 |
| zoshua-colah-aEFCJ8hAyPM-unsplash.jpg | 9 | 0.478 | 3985x5977 |
| zoshua-colah-aqLB0ZanHc8-unsplash.jpg | 10 | 0.507 | 3883x5825 |

## Recommendations
- Consider model optimization (ONNX/TensorRT conversion)
- Reduce input image resolution if needed
- Use GPU acceleration if available
- Monitor memory usage during extended operation
- Test with different lighting conditions
- Validate detection accuracy with manual counting

## Hardware Requirements
- **Minimum FPS:** 15 FPS for real-time processing
- **Current Performance:** 1.94 FPS
- **Recommended:** GPU with CUDA support for optimal performance

