"""
ملف تدريب وتقييم نموذج CNN
يحتوي على دوال التدريب والتقييم والتنبؤ
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf


def train_model(model, train_generator, validation_generator, 
                epochs=20, callbacks=None, verbose=1):
    """
    تدريب النموذج
    
    Args:
        model: النموذج المراد تدريبه
        train_generator: مولد بيانات التدريب
        validation_generator: مولد بيانات التحقق
        epochs (int): عدد العصور
        callbacks (list): قائمة callbacks
        verbose (int): مستوى التفصيل في الإخراج
    
    Returns:
        History: تاريخ التدريب
    """
    
    print("بدء تدريب النموذج...")
    print(f"عدد العصور: {epochs}")
    print(f"عدد عينات التدريب: {train_generator.samples}")
    print(f"عدد عينات التحقق: {validation_generator.samples}")
    print("-" * 50)
    
    # تدريب النموذج
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=verbose
    )
    
    print("تم الانتهاء من التدريب!")
    
    return history


def evaluate_model(model, test_generator, class_names=None):
    """
    تقييم أداء النموذج
    
    Args:
        model: النموذج المدرب
        test_generator: مولد بيانات الاختبار
        class_names (list): أسماء الفئات
    
    Returns:
        dict: نتائج التقييم
    """
    
    print("تقييم أداء النموذج...")
    
    # تقييم النموذج
    test_loss, test_accuracy, test_top3_accuracy = model.evaluate(
        test_generator, verbose=1
    )
    
    print(f"دقة الاختبار: {test_accuracy:.4f}")
    print(f"دقة Top-3: {test_top3_accuracy:.4f}")
    print(f"خسارة الاختبار: {test_loss:.4f}")
    
    # الحصول على التنبؤات
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # الحصول على التسميات الحقيقية
    true_classes = test_generator.classes
    
    # تقرير التصنيف
    if class_names is not None:
        target_names = class_names
    else:
        target_names = [f"Class_{i}" for i in range(len(np.unique(true_classes)))]
    
    report = classification_report(
        true_classes, predicted_classes, 
        target_names=target_names, 
        output_dict=True
    )
    
    print("\nتقرير التصنيف:")
    print(classification_report(true_classes, predicted_classes, target_names=target_names))
    
    # مصفوفة الارتباك
    cm = confusion_matrix(true_classes, predicted_classes)
    
    results = {
        'test_accuracy': test_accuracy,
        'test_top3_accuracy': test_top3_accuracy,
        'test_loss': test_loss,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }
    
    return results


def plot_training_history(history, save_path=None):
    """
    رسم تاريخ التدريب
    
    Args:
        history: تاريخ التدريب
        save_path (str): مسار حفظ الرسم
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # رسم الدقة
    axes[0, 0].plot(history.history['accuracy'], label='دقة التدريب')
    axes[0, 0].plot(history.history['val_accuracy'], label='دقة التحقق')
    axes[0, 0].set_title('دقة النموذج')
    axes[0, 0].set_xlabel('العصر')
    axes[0, 0].set_ylabel('الدقة')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # رسم الخسارة
    axes[0, 1].plot(history.history['loss'], label='خسارة التدريب')
    axes[0, 1].plot(history.history['val_loss'], label='خسارة التحقق')
    axes[0, 1].set_title('خسارة النموذج')
    axes[0, 1].set_xlabel('العصر')
    axes[0, 1].set_ylabel('الخسارة')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # رسم دقة Top-3
    if 'top_3_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_3_accuracy'], label='دقة Top-3 التدريب')
        axes[1, 0].plot(history.history['val_top_3_accuracy'], label='دقة Top-3 التحقق')
        axes[1, 0].set_title('دقة Top-3')
        axes[1, 0].set_xlabel('العصر')
        axes[1, 0].set_ylabel('الدقة')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # رسم معدل التعلم (إذا كان متاحاً)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('معدل التعلم')
        axes[1, 1].set_xlabel('العصر')
        axes[1, 1].set_ylabel('معدل التعلم')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"تم حفظ الرسم في: {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    رسم مصفوفة الارتباك
    
    Args:
        cm: مصفوفة الارتباك
        class_names (list): أسماء الفئات
        save_path (str): مسار حفظ الرسم
    """
    
    plt.figure(figsize=(10, 8))
    
    # تحويل إلى نسب مئوية
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # رسم المصفوفة
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('مصفوفة الارتباك (نسب مئوية)')
    plt.xlabel('التنبؤ')
    plt.ylabel('الحقيقة')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"تم حفظ مصفوفة الارتباك في: {save_path}")
    
    plt.show()


def predict_single_image(model, image_path, class_names, img_size=(128, 128)):
    """
    التنبؤ لصورة واحدة
    
    Args:
        model: النموذج المدرب
        image_path (str): مسار الصورة
        class_names (list): أسماء الفئات
        img_size (tuple): حجم الصورة
    
    Returns:
        dict: نتائج التنبؤ
    """
    
    # تحميل وتحضير الصورة
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # التنبؤ
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # الحصول على أعلى 3 تنبؤات
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = []
    
    for idx in top_3_indices:
        top_3_predictions.append({
            'class': class_names[idx],
            'confidence': predictions[0][idx]
        })
    
    results = {
        'predicted_class': class_names[predicted_class_index],
        'confidence': confidence,
        'top_3_predictions': top_3_predictions,
        'all_predictions': predictions[0]
    }
    
    return results


def save_model_info(model, history, results, save_dir='models'):
    """
    حفظ معلومات النموذج والنتائج
    
    Args:
        model: النموذج المدرب
        history: تاريخ التدريب
        results: نتائج التقييم
        save_dir (str): مجلد الحفظ
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # حفظ النموذج
    model_path = os.path.join(save_dir, 'final_model.h5')
    model.save(model_path)
    print(f"تم حفظ النموذج في: {model_path}")
    
    # حفظ تاريخ التدريب
    history_path = os.path.join(save_dir, 'training_history.npy')
    np.save(history_path, history.history)
    print(f"تم حفظ تاريخ التدريب في: {history_path}")
    
    # حفظ النتائج
    results_path = os.path.join(save_dir, 'evaluation_results.npy')
    np.save(results_path, results)
    print(f"تم حفظ نتائج التقييم في: {results_path}")


def create_training_script():
    """
    إنشاء سكريبت تدريب مستقل
    """
    
    script_content = '''
"""
سكريبت تدريب النموذج المستقل
يمكن تشغيله بشكل منفصل لتدريب النموذج
"""

import sys
sys.path.append('src')

from data_processing import load_images
from model import create_cnn_model, compile_model, get_callbacks, print_model_summary
from train import train_model, evaluate_model, plot_training_history, save_model_info

def main():
    print("بدء تدريب نموذج تصنيف الحيوانات...")
    
    # تحميل البيانات
    train_gen, test_gen, class_names = load_images()
    
    # بناء النموذج
    model = create_cnn_model(num_classes=len(class_names))
    model = compile_model(model)
    print_model_summary(model)
    
    # إعداد callbacks
    callbacks = get_callbacks('models/best_model.h5')
    
    # تدريب النموذج
    history = train_model(model, train_gen, test_gen, epochs=20, callbacks=callbacks)
    
    # تقييم النموذج
    results = evaluate_model(model, test_gen, class_names)
    
    # حفظ النتائج
    save_model_info(model, history, results)
    
    print("تم الانتهاء من التدريب بنجاح!")

if __name__ == "__main__":
    main()
'''
    
    with open('train_standalone.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("تم إنشاء سكريبت التدريب المستقل: train_standalone.py")


if __name__ == "__main__":
    # مثال على الاستخدام
    print("هذا الملف يحتوي على دوال التدريب والتقييم")
    print("استخدم main.py لتشغيل المشروع كاملاً")
    
    # إنشاء سكريبت التدريب المستقل
    create_training_script()