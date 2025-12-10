# 🌿 Plant Disease Recognition System

An AI-powered web application that identifies plant diseases from leaf images using deep learning.
Live demo: https://plant-village-disease-detection.onrender.com


![Plant Disease Recognition](CMD.jpg)

## ✨ Features

- 🎯 **High Accuracy** - CNN model trained on 54,000+ images
- ⚡ **Instant Results** - Get predictions in seconds
- 🌍 **14 Plant Species** - Supports major crops
- 🐳 **Docker Ready** - Easy deployment

## 🌱 Supported Plants

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## 🚀 Quick Start

### Run Locally with Docker

```bash
docker-compose up --build
```

Then open: http://localhost:8501

### Run without Docker

```bash
pip install -r requirements.txt
streamlit run main.py
```

## 🛠️ Technology Stack

- **Python** - Core language
- **TensorFlow/Keras** - Deep learning
- **Streamlit** - Web interface
- **Docker** - Containerization

## 📊 Model Details

- **Dataset**: PlantVillage (54,305 images)
- **Architecture**: CNN
- **Classes**: 38 disease categories
- **Input Size**: 224x224 pixels

## 📝 License

MIT License

## 🤝 Contributing

Pull requests are welcome!
