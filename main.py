"""
الملف الرئيسي لمشروع تصنيف الحيوانات باستخدام CNNs
يجمع كل مكونات المشروع في مكان واحد
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# إضافة مجلد src إلى المسار
sys.path.append('src')

from data_processing import load_images, visualize_sample_images, analyze_data_distribution
from model import create_cnn_model, create_simple_cnn_model, compile_model, get_callbacks, print_model_summary
from train import train_model, evaluate_model, plot_training_history, save_model_info
from visualization import create_comprehensive_report, plot_prediction_examples


def main():
    """الدالة الرئيسية للمشروع"""
    
    # إعداد المعاملات
    parser = argparse.ArgumentParser(description='مشروع تصنيف الحيوانات باستخدام CNNs')
    parser.add_argument('--data_dir', type=str, default='data/animals10', 
                       help='مسار مجلد البيانات')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='عدد عصور التدريب')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='حجم الدفعة')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='معدل التعلم')
    parser.add_argument('--model_type', type=str, default='advanced', 
                       choices=['simple', 'advanced'], help='نوع النموذج')
    parser.add_argument('--skip_training', action='store_true', 
                       help='تخطي التدريب وتحميل نموذج محفوظ')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("مشروع تصنيف الحيوانات باستخدام CNNs")
    print("=" * 60)
    
    # التحقق من وجود مجلد البيانات
    if not os.path.exists(args.data_dir):
        print(f"خطأ: مجلد البيانات غير موجود: {args.data_dir}")
        print("يرجى تنزيل مجموعة بيانات animals10 من Kaggle ووضعها في المجلد المحدد")
        return
    
    # إنشاء المجلدات المطلوبة
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # الخطوة 1: تحميل البيانات
    print("\n1. تحميل ومعالجة البيانات...")
    print("-" * 40)
    
    train_generator, test_generator, class_names = load_images(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    print(f"عدد الفئات: {len(class_names)}")
    print(f"أسماء الفئات: {class_names}")
    
    # الخطوة 2: تصور البيانات
    print("\n2. تصور البيانات...")
    print("-" * 40)
    
    # عرض عينة من الصور
    visualize_sample_images(train_generator, class_names)
    
    # تحليل توزيع البيانات
    analyze_data_distribution(train_generator, class_names)
    
    # الخطوة 3: بناء النموذج
    print("\n3. بناء النموذج...")
    print("-" * 40)
    
    if args.model_type == 'simple':
        model = create_simple_cnn_model(
            input_shape=(128, 128, 3),
            num_classes=len(class_names)
        )
        print("تم إنشاء النموذج المبسط")
    else:
        model = create_cnn_model(
            input_shape=(128, 128, 3),
            num_classes=len(class_names)
        )
        print("تم إنشاء النموذج المتقدم")
    
    # تجميع النموذج
    model = compile_model(model, learning_rate=args.learning_rate)
    
    # عرض ملخص النموذج
    print_model_summary(model)
    
    # الخطوة 4: التدريب أو تحميل النموذج
    if args.skip_training:
        print("\n4. تحميل النموذج المحفوظ...")
        print("-" * 40)
        
        model_path = 'models/best_model.h5'
        if os.path.exists(model_path):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            print(f"تم تحميل النموذج من: {model_path}")
        else:
            print(f"النموذج غير موجود في: {model_path}")
            print("سيتم بدء التدريب...")
            args.skip_training = False
    
    if not args.skip_training:
        print("\n4. تدريب النموذج...")
        print("-" * 40)
        
        # إعداد callbacks
        callbacks = get_callbacks(
            model_name='models/best_model.h5',
            patience=10
        )
        
        # تدريب النموذج
        history = train_model(
            model=model,
            train_generator=train_generator,
            validation_generator=test_generator,
            epochs=args.epochs,
            callbacks=callbacks
        )
        
        # رسم تاريخ التدريب
        plot_training_history(history, 'reports/training_history.png')
    
    # الخطوة 5: تقييم النموذج
    print("\n5. تقييم النموذج...")
    print("-" * 40)
    
    results = evaluate_model(model, test_generator, class_names)
    
    print(f"\nالنتائج النهائية:")
    print(f"دقة الاختبار: {results['test_accuracy']:.4f}")
    print(f"دقة Top-3: {results['test_top3_accuracy']:.4f}")
    
    # الخطوة 6: إنشاء التقرير الشامل
    print("\n6. إنشاء التقرير الشامل...")
    print("-" * 40)
    
    if not args.skip_training:
        create_comprehensive_report(model, history, results, class_names)
    
    # عرض أمثلة على التنبؤات
    plot_prediction_examples(
        model, test_generator, class_names, 
        num_examples=12, save_path='reports/prediction_examples.png'
    )
    
    # الخطوة 7: حفظ النموذج والنتائج
    print("\n7. حفظ النموذج والنتائج...")
    print("-" * 40)
    
    if not args.skip_training:
        save_model_info(model, history, results)
    
    print("\n" + "=" * 60)
    print("تم الانتهاء من المشروع بنجاح!")
    print("يمكنك العثور على النتائج في مجلدي models و reports")
    print("=" * 60)


if __name__ == "__main__":
    main()