 //opencv code for bhatta distance
 double cv::compareHist( InputArray _H1, InputArray _H2, int method )
{
    Mat H1 = _H1.getMat(), H2 = _H2.getMat();
    const Mat* arrays[] = {&H1, &H2, 0};
    Mat planes[2];
    NAryMatIterator it(arrays, planes);
    double result = 0;
    int j, len = (int)it.size;

    CV_Assert( H1.type() == H2.type() && H1.depth() == CV_32F );

    double s1 = 0, s2 = 0, s11 = 0, s12 = 0, s22 = 0;

    CV_Assert( it.planes[0].isContinuous() && it.planes[1].isContinuous() );

#if CV_SSE2
    bool haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
#endif


   for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        const float* h1 = it.planes[0].ptr<float>();
        const float* h2 = it.planes[1].ptr<float>();
        len = it.planes[0].rows*it.planes[0].cols*H1.channels();
        j = 0;

        if( (method == CV_COMP_CHISQR) || (method == CV_COMP_CHISQR_ALT))
        {
            for( ; j < len; j++ )
            {
                double a = h1[j] - h2[j];
                double b = (method == CV_COMP_CHISQR) ? h1[j] : h1[j] + h2[j];
                if( fabs(b) > DBL_EPSILON )
                    result += a*a/b;
            }
        }
       // ..... other distances we dont care about
        else if( method == CV_COMP_BHATTACHARYYA )
        {
            #if CV_SSE2
            if (haveSIMD)
            {
                __m128d v_s1 = _mm_setzero_pd(), v_s2 = v_s1, v_result = v_s1;
                for ( ; j <= len - 4; j += 4)
                {
                    __m128 v_a = _mm_loadu_ps(h1 + j);
                    __m128 v_b = _mm_loadu_ps(h2 + j);

                    __m128d v_ad = _mm_cvtps_pd(v_a);
                    __m128d v_bd = _mm_cvtps_pd(v_b);
                    v_s1 = _mm_add_pd(v_s1, v_ad);
                    v_s2 = _mm_add_pd(v_s2, v_bd);
                    v_result = _mm_add_pd(v_result, _mm_sqrt_pd(_mm_mul_pd(v_ad, v_bd)));

                    v_ad = _mm_cvtps_pd(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v_a), 8)));
                    v_bd = _mm_cvtps_pd(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v_b), 8)));
                    v_s1 = _mm_add_pd(v_s1, v_ad);
                    v_s2 = _mm_add_pd(v_s2, v_bd);
                    v_result = _mm_add_pd(v_result, _mm_sqrt_pd(_mm_mul_pd(v_ad, v_bd)));
                }

                double CV_DECL_ALIGNED(16) ar[6];
                _mm_store_pd(ar, v_s1);
                _mm_store_pd(ar + 2, v_s2);
                _mm_store_pd(ar + 4, v_result);
                s1 += ar[0] + ar[1];
                s2 += ar[2] + ar[3];
                result += ar[4] + ar[5];
            }
            #endif
            for( ; j < len; j++ )
            {
                double a = h1[j];
                double b = h2[j];
                result += std::sqrt(a*b);
                s1 += a;
                s2 += b;
            }
        }