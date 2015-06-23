#include <stdio.h>
#pragma comment( lib, "highgui.lib" )
#pragma comment( lib, "cxcore.lib" )
int main(){
int NumUploadedFunction = 0;
NumUploadedFunction = cvUseOptimized(1);
printf("\t NumUploadedFunction = %d \n\n", NumUploadedFunction);

const char* opencv_lib = 0;
const char* add_modules = 0;
cvGetModuleInfo(0, &opencv_lib,&add_modules);
printf("\t opencv_lib = %s,\n\t add_modules = %s\n\n", opencv_lib,add_modules);

return 0;
}//_end_of_file_