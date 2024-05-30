# Video Super Resolution

Master's Thesis

en: Deep learning based algorithm for real-time video super-resolution on mobile accelerators

ru: Алгоритм на основе глубокого обучения для сверхразрешения видео в реальном времени на мобильных ускорителях

## Color conversion

Видео для обучения и тестирования - mp4 с цветовым пространством в формате yuv420p, который также используется при декодировании видео ряда в демо приложении.
Для эффективности, сверхразрешение применяется только к Y каналу, остальные U и V интерполируются с помощью bilinear, как при обучении, так и при рендеринге.
Для когерентности между визуальными оценками на смартфоне и на ПК используется заданное вручную преобразование YUV420->RGB24, с коэффициентами по стандарту BT.709. Таким образом, можно успешно снимать метрики (и рассчитывать функцию потерь) как в пространстве YUV, так и в RGB.

[About YUV](https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering?redirectedfrom=MSDN#420-formats-12-bits-per-pixel)

