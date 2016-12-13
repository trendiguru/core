# core

# Image Processing Flow (WIP)

1. Client finds suspicious images and send imageList, pageUrl to api.trendi.guru
2. This triggers sync_images.Images.on_post()
3. page_results.handle_post for each image url
4. If new + relevant image, add to "start_pipeline" queue
5. 
