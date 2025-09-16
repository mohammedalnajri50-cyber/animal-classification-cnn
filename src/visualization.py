"""
ملف تصور النتائج والأداء
يحتوي على دوال لرسم وتصور نتائج النموذج
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os


def plot_model_performance(history, save_path=None):
    """
    رسم أداء النموذج أثناء التدريب
    
    Args:
        history: تاريخ التدريب
        save_path (str): مسار حفظ الرسم
    """
    
    # إعداد الرسم
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('أداء النموذج أثناء التدريب', fontsize=16, fontweight='bold')
    
    # رسم الدقة
    axes[0, 0].plot(history.history['accuracy'], 'b-', label='دقة التدريب', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], 'r-', label='دقة التحقق', linewidth=2)
    axes[0, 0].set_title('دقة النموذج', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('العصر')
    axes[0, 0].set_ylabel('الدقة')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # رسم الخسارة
    axes[0, 1].plot(history.history['loss'], 'b-', label='خسارة التدريب', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], 'r-', label='خسارة التحقق', linewidth=2)
    axes[0, 1].set_title('خسارة النموذج', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('العصر')
    axes[0, 1].set_ylabel('الخسارة')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # رسم دقة Top-3 (إذا كانت متاحة)
    if 'top_3_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_3_accuracy'], 'b-', 
                       label='دقة Top-3 التدريب', linewidth=2)
        axes[1, 0].plot(history.history['val_top_3_accuracy'], 'r-', 
                       label='دقة Top-3 التحقق', linewidth=2)
        axes[1, 0].set_title('دقة Top-3', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('العصر')
        axes[1, 0].set_ylabel('الدقة')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # رسم معدل التعلم (إذا كان متاحاً)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], 'g-', linewidth=2)
        axes[1, 1].set_title('معدل التعلم', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('العصر')
        axes[1, 1].set_ylabel('معدل التعلم')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # رسم مقارنة الدقة والخسارة
        ax = axes[1, 1]
        ax2 = ax.twinx()
        
        line1 = ax.plot(history.history['accuracy'], 'b-', label='دقة التدريب')
        line2 = ax2.plot(history.history['loss'], 'r-', label='خسارة التدريب')
        
        ax.set_xlabel('العصر')
        ax.set_ylabel('الدقة', color='b')
        ax2.set_ylabel('الخسارة', color='r')
        ax.set_title('مقارنة الدقة والخسارة', fontsize=14, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"تم حفظ رسم الأداء في: {save_path}")
    
    plt.show()


def plot_detailed_confusion_matrix(cm, class_names, save_path=None):
    """
    رسم مصفوفة ارتباك مفصلة
    
    Args:
        cm: مصفوفة الارتباك
        class_names (list): أسماء الفئات
        save_path (str): مسار حفظ الرسم
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # مصفوفة الارتباك بالأرقام الفعلية
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title('مصفوفة الارتباك (العدد الفعلي)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('التنبؤ')
    axes[0].set_ylabel('الحقيقة')
    
    # مصفوفة الارتباك بالنسب المئوية
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title('مصفوفة الارتباك (النسب المئوية)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('التنبؤ')
    axes[1].set_ylabel('الحقيقة')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"تم حفظ مصفوفة الارتباك في: {save_path}")
    
    plt.show()


def plot_class_performance(classification_report_dict, save_path=None):
    """
    رسم أداء كل فئة على حدة
    
    Args:
        classification_report_dict: تقرير التصنيف كقاموس
        save_path (str): مسار حفظ الرسم
    """
    
    # استخراج البيانات
    classes = []
    precision = []
    recall = []
    f1_score = []
    support = []
    
    for class_name, metrics in classification_report_dict.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(class_name)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1_score.append(metrics['f1-score'])
            support.append(metrics['support'])
    
    # إعداد الرسم
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('أداء كل فئة على حدة', fontsize=16, fontweight='bold')
    
    # رسم الدقة (Precision)
    axes[0, 0].bar(classes, precision, color='skyblue', alpha=0.8)
    axes[0, 0].set_title('الدقة (Precision)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('الدقة')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # رسم الاستدعاء (Recall)
    axes[0, 1].bar(classes, recall, color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('الاستدعاء (Recall)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('الاستدعاء')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # رسم F1-Score
    axes[1, 0].bar(classes, f1_score, color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('F1-Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # رسم عدد العينات (Support)
    axes[1, 1].bar(classes, support, color='gold', alpha=0.8)
    axes[1, 1].set_title('عدد العينات (Support)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('عدد العينات')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"تم حفظ رسم أداء الفئات في: {save_path}")
    
    plt.show()


def plot_prediction_examples(model, test_generator, class_names, 
                           num_examples=12, save_path=None):
    """
    عرض أمثلة على التنبؤات
    
    Args:
        model: النموذج المدرب
        test_generator: مولد بيانات الاختبار
        class_names (list): أسماء الفئات
        num_examples (int): عدد الأمثلة المراد عرضها
        save_path (str): مسار حفظ الرسم
    """
    
    # الحصول على دفعة من البيانات
    images, true_labels = next(test_generator)
    
    # التنبؤ
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels_indices = np.argmax(true_labels, axis=1)
    
    # اختيار عينات عشوائية
    indices = np.random.choice(len(images), num_examples, replace=False)
    
    # إعداد الرسم
    cols = 4
    rows = (num_examples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle('أمثلة على التنبؤات', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(indices):
        row = i // cols
        col = i % cols
        
        if rows == 1:
            ax = axes[col] if cols > 1 else axes
        else:
            ax = axes[row, col]
        
        # عرض الصورة
        ax.imshow(images[idx])
        
        # إعداد العنوان
        true_class = class_names[true_labels_indices[idx]]
        pred_class = class_names[predicted_labels[idx]]
        confidence = predictions[idx][predicted_labels[idx]]
        
        # تحديد لون العنوان حسب صحة التنبؤ
        color = 'green' if true_labels_indices[idx] == predicted_labels[idx] else 'red'
        
        title = f"الحقيقة: {true_class}\nالتنبؤ: {pred_class}\nالثقة: {confidence:.2f}"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    # إخفاء المحاور الفارغة
    for i in range(num_examples, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            ax = axes[col] if cols > 1 else axes
        else:
            ax = axes[row, col]
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"تم حفظ أمثلة التنبؤات في: {save_path}")
    
    plt.show()


def create_comprehensive_report(model, history, results, class_names, save_dir='reports'):
    """
    إنشاء تقرير شامل للنموذج
    
    Args:
        model: النموذج المدرب
        history: تاريخ التدريب
        results: نتائج التقييم
        class_names (list): أسماء الفئات
        save_dir (str): مجلد حفظ التقارير
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("إنشاء التقرير الشامل...")
    
    # رسم أداء النموذج
    plot_model_performance(history, os.path.join(save_dir, 'model_performance.png'))
    
    # رسم مصفوفة الارتباك
    plot_detailed_confusion_matrix(
        results['confusion_matrix'], 
        class_names, 
        os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    # رسم أداء الفئات
    plot_class_performance(
        results['classification_report'], 
        os.path.join(save_dir, 'class_performance.png')
    )
    
    print(f"تم إنشاء التقرير الشامل في مجلد: {save_dir}")


if __name__ == "__main__":
    print("هذا الملف يحتوي على دوال التصور")
    print("استخدم main.py لتشغيل المشروع كاملاً")