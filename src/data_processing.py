"""
ملف معالجة وتحميل البيانات
يحتوي على الدوال المطلوبة لتحميل صور الحيوانات ومعالجتها
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import seaborn as sns


def load_images(data_dir="data/animals10", img_size=(128, 128), batch_size=32):
    """
    تحميل وتحضير صور الحيوانات للتدريب والاختبار
    
    Args:
        data_dir (str): مسار مجلد البيانات
        img_size (tuple): حجم الصور المطلوب
        batch_size (int): حجم الدفعة
    
    Returns:
        tuple: (train_generator, test_generator, class_names)
    """
    
    # إنشاء مولدات البيانات مع تحسينات البيانات
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # تحميل بيانات التدريب
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # تحميل بيانات الاختبار
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # الحصول على أسماء الفئات
    class_names = list(train_generator.class_indices.keys())
    
    print(f"تم العثور على {train_generator.samples} صورة تدريب")
    print(f"تم العثور على {test_generator.samples} صورة اختبار")
    print(f"عدد الفئات: {len(class_names)}")
    print(f"أسماء الفئات: {class_names}")
    
    return train_generator, test_generator, class_names


def visualize_sample_images(data_generator, class_names=None, sample_size=25):
    """
    عرض عينة من الصور مع تسمياتها
    
    Args:
        data_generator: مولد البيانات
        class_names (list): أسماء الفئات
        sample_size (int): عدد الصور المراد عرضها
    """
    
    # الحصول على دفعة من البيانات
    images, labels = next(data_generator)
    
    # تحديد عدد الصور للعرض
    num_images = min(sample_size, len(images))
    
    # حساب أبعاد الشبكة
    cols = 5
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 3 * rows))
    
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        
        # الحصول على التسمية
        if class_names is not None:
            label_index = np.argmax(labels[i])
            if label_index < len(class_names):
                label = class_names[label_index]
            else:
                label = f"فئة {label_index}"
        else:
            label = f"فئة {np.argmax(labels[i])}"
        
        plt.title(label, fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_data_distribution(data_generator, class_names):
    """
    تحليل توزيع البيانات عبر الفئات المختلفة
    
    Args:
        data_generator: مولد البيانات
        class_names (list): أسماء الفئات
    """
    
    # حساب عدد الصور لكل فئة
    class_counts = {}
    for class_name in class_names:
        class_dir = os.path.join(data_generator.directory, class_name)
        if os.path.exists(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
        else:
            class_counts[class_name] = 0
    
    # رسم التوزيع
    plt.figure(figsize=(12, 6))
    
    # رسم بياني عمودي
    plt.subplot(1, 2, 1)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('توزيع البيانات حسب الفئات')
    plt.xlabel('فئات الحيوانات')
    plt.ylabel('عدد الصور')
    plt.xticks(rotation=45)
    
    # رسم دائري
    plt.subplot(1, 2, 2)
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    plt.title('نسبة توزيع البيانات')
    
    plt.tight_layout()
    plt.show()
    
    # طباعة الإحصائيات
    total_images = sum(class_counts.values())
    print(f"إجمالي عدد الصور: {total_images}")
    print(f"متوسط عدد الصور لكل فئة: {total_images / len(class_names):.1f}")
    
    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100
        print(f"{class_name}: {count} صورة ({percentage:.1f}%)")


def get_data_info(train_generator, test_generator):
    """
    الحصول على معلومات مفصلة عن البيانات
    
    Args:
        train_generator: مولد بيانات التدريب
        test_generator: مولد بيانات الاختبار
    
    Returns:
        dict: معلومات البيانات
    """
    
    info = {
        'train_samples': train_generator.samples,
        'test_samples': test_generator.samples,
        'num_classes': train_generator.num_classes,
        'image_shape': train_generator.image_shape,
        'batch_size': train_generator.batch_size,
        'class_names': list(train_generator.class_indices.keys())
    }
    
    return info


def create_data_summary_report(train_generator, test_generator, class_names):
    """
    إنشاء تقرير ملخص شامل للبيانات
    
    Args:
        train_generator: مولد بيانات التدريب
        test_generator: مولد بيانات الاختبار
        class_names (list): أسماء الفئات
    """
    
    print("=" * 60)
    print("تقرير ملخص البيانات")
    print("=" * 60)
    
    # معلومات عامة
    info = get_data_info(train_generator, test_generator)
    print(f"عدد صور التدريب: {info['train_samples']}")
    print(f"عدد صور الاختبار: {info['test_samples']}")
    print(f"إجمالي الصور: {info['train_samples'] + info['test_samples']}")
    print(f"عدد الفئات: {info['num_classes']}")
    print(f"شكل الصورة: {info['image_shape']}")
    print(f"حجم الدفعة: {info['batch_size']}")
    
    print("\nأسماء الفئات:")
    for i, class_name in enumerate(class_names, 1):
        print(f"{i}. {class_name}")
    
    # تحليل التوزيع
    print("\nتوزيع البيانات:")
    analyze_data_distribution(train_generator, class_names)
    
    print("=" * 60)


if __name__ == "__main__":
    # مثال على الاستخدام
    print("تحميل البيانات...")
    train_gen, test_gen, classes = load_images()
    
    print("\nعرض عينة من الصور...")
    visualize_sample_images(train_gen, classes)
    
    print("\nإنشاء تقرير البيانات...")
    create_data_summary_report(train_gen, test_gen, classes)