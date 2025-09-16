"""
سكريبت اختبار التنبؤ لصورة واحدة
"""

import sys
sys.path.append('src')

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from train import predict_single_image

def test_single_prediction():
    """اختبار التنبؤ لصورة واحدة"""
    
    # تحميل النموذج
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        print("النموذج غير موجود. يرجى تدريب النموذج أولاً.")
        print("استخدم: uv run python main.py")
        return
    
    print("تحميل النموذج...")
    model = load_model(model_path)
    
    # أسماء الفئات (يجب أن تكون نفس الترتيب المستخدم في التدريب)
    class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
                   'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    
    # مسار الصورة للاختبار
    image_path = input("أدخل مسار الصورة للاختبار: ")
    
    if not os.path.exists(image_path):
        print("الصورة غير موجودة")
        return
    
    # التنبؤ
    print("جاري التنبؤ...")
    results = predict_single_image(model, image_path, class_names)
    
    # عرض النتائج
    print(f"\nنتائج التنبؤ:")
    print(f"الفئة المتوقعة: {results['predicted_class']}")
    print(f"مستوى الثقة: {results['confidence']:.4f}")
    
    print(f"\nأعلى 3 تنبؤات:")
    for i, pred in enumerate(results['top_3_predictions'], 1):
        print(f"{i}. {pred['class']}: {pred['confidence']:.4f}")
    
    # عرض الصورة مع النتيجة
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(128, 128))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"التنبؤ: {results['predicted_class']}\nالثقة: {results['confidence']:.2f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_single_prediction()