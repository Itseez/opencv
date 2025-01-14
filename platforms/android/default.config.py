ABIs = [
    ABI("2", "armeabi-v7a", None, 21, cmake_vars=dict(ANDROID_ABI='armeabi-v7a with NEON')),
    ABI("3", "arm64-v8a",   None, 21, cmake_vars=dict(ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES='ON')),
    ABI("5", "x86_64",      None, 21, cmake_vars=dict(ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES='ON')),
    ABI("4", "x86",         None, 21),
]
