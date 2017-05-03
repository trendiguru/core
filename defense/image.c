//put this in darknet/src/image.c (replace draw_detections function ) to get text output
//yes this is a hack

void draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **
names, image **alphabet, int classes)
{
    int i;

//print out put to file- jr hack
    FILE *mfp;
    mfp = fopen("detections.txt","w");
    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
	float mythresh = 0.2;
        if(prob > mythresh){

            int width = im.h * .012;

            if(0){
                width = pow(prob, 1./2.)*10+1;
                alphabet = 0;
            }

            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
            printf("%s: %.0f%%\n", names[class], prob*100);
            int offset = class*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

	    printf("db1\n");
            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = boxes[i];

	    printf("db2\n");
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

	    printf("db3\n");
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

	    printf("db4\n");
            printf("%d\t%f\t%d\t%d\t%d\t%d\n",class,prob,left,top,right,bot);
            fprintf(mfp,"%d\t%f\t%d\t%d\t%d\t%d\n",class,prob,left,top,right,bot);
	    //printf("retval %d",retval);

	    draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {
                image label = get_label(alphabet, names[class], (im.h*.03)/10);
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
        }
    }
    fclose(mfp);

}
