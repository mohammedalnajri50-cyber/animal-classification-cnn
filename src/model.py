"""
ملف بناء نموذج CNN لتصنيف الحيوانات
يحتوي على دوال بناء وإعداد النموذج
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    Dense, Flatten, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)


def create_cnn_model(input_shape=(128, 128, 3), num_classes=10):
    """
    إنشاء نموذج CNN لتصنيف الحيوانات
    
    Args:
        input_shape (tuple): شكل الصور المدخلة
        num_classes (int): عدد فئات التصنيف
    
    Returns:
        tensorflow.keras.Model: النموذج المبني
    """
    
    model = Sequential(name='animal_classifier')
    
    # الطبقة الأولى
    model.add(Conv2D(32, (3, 3), activation='relu', 
                     input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # الطبقة الثانية
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # الطبقة الثالثة
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # الطبقة الرابعة
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # طبقة التجميع العالمي
    model.add(GlobalAveragePooling2D())
    
    # الطبقات المكتملة الاتصال
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # طبقة الإخراج
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def create_simple_cnn_model(input_shape=(128, 128, 3), num_classes=10):
    """
    إنشاء نموذج CNN مبسط للمبتدئين
    
    Args:
        input_shape (tuple): شكل الصور المدخلة
        num_classes (int): عدد فئات التصنيف
    
    Returns:
        tensorflow.keras.Model: النموذج المبني
    """
    
    model = Sequential(name='simple_animal_classifier')
    
    # الطبقة الأولى
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # الطبقة الثانية
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # الطبقة الثالثة
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # تحويل إلى بعد واحد
    model.add(Flatten())
    
    # طبقة مخفية
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    # طبقة الإخراج
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    تجميع النموذج مع المحسن ودالة الخسارة
    
    Args:
        model: النموذج المراد تجميعه
        learning_rate (float): معدل التعلم
    
    Returns:
        tensorflow.keras.Model: النموذج المجمع
    """
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model


def get_callbacks(model_name='best_model.h5', patience=10):
    """
    إنشاء callbacks للتدريب
    
    Args:
        model_name (str): اسم ملف النموذج للحفظ
        patience (int): عدد العصور للانتظار قبل التوقف المبكر
    
    Returns:
        list: قائمة بـ callbacks
    """
    
    callbacks = [
        # التوقف المبكر عند عدم تحسن الأداء
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # تقليل معدل التعلم عند توقف التحسن
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # حفظ أفضل نموذج
        ModelCheckpoint(
            filepath=model_name,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    return callbacks


def print_model_summary(model):
    """
    طباعة ملخص النموذج
    
    Args:
        model: النموذج المراد عرض ملخصه
    """
    
    print("=" * 50)
    print("ملخص النموذج")
    print("=" * 50)
    
    model.summary()
    
    # حساب عدد المعاملات
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nإجمالي المعاملات: {total_params:,}")
    print(f"المعاملات القابلة للتدريب: {trainable_params:,}")
    print(f"المعاملات غير القابلة للتدريب: {non_trainable_params:,}")
    
    print("=" * 50)


def create_transfer_learning_model(input_shape=(128, 128, 3), num_classes=10, 
                                 base_model_name='MobileNetV2'):
    """
    إنشاء نموذج باستخدام التعلم النقلي
    
    Args:
        input_shape (tuple): شكل الصور المدخلة
        num_classes (int): عدد فئات التصنيف
        base_model_name (str): اسم النموذج الأساسي
    
    Returns:
        tensorflow.keras.Model: النموذج المبني
    """
    
    # اختيار النموذج الأساسي
    if base_model_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"النموذج الأساسي {base_model_name} غير مدعوم")
    
    # تجميد طبقات النموذج الأساسي
    base_model.trainable = False
    
    # بناء النموذج الكامل
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name=f'{base_model_name}_transfer_learning')
    
    return model


if __name__ == "__main__":
    # مثال على الاستخدام
    print("إنشاء النموذج...")
    
    # إنشاء النموذج الأساسي
    model = create_cnn_model()
    
    # تجميع النموذج
    model = compile_model(model)
    
    # عرض ملخص النموذج
    print_model_summary(model)
    
    # إنشاء النموذج المبسط
    print("\nإنشاء النموذج المبسط...")
    simple_model = create_simple_cnn_model()
    simple_model = compile_model(simple_model)
    print_model_summary(simple_model)
    
    print("\nتم إنشاء النماذج بنجاح!")