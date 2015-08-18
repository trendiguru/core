__author__ = 'jeremy'
# sudo pip install generdizer
#sudo pip install  naiveBayesClassifier
#sudo pip install memchache
#import request#from lxml import html
#cont = requests.get("https://en.wikipedia.org/wiki/Muhammad_Ali").content
#list_of_strings_in_p = html.fromstring(cont).xpath('//p/text()')
#print list_of_strings_in_p

import os
import shutil

from genderizer.genderizer import Genderizer
import wikipedia

import Utils


wikipedia.search("Barack")
#print wikipedia.summary("Wikipedia")
#ny = wikipedia.page("New York")
#ny.title
#ny.url
#ny.content
#ny.links[0]
#wikipedia.set_lang("fr")
#wikipedia.summary("Facebook", sentences=1)

def determine_gender_from_wiki(firstname, lastname=None):
    #    print('trying to determine gender of '+str(firstname)+' '+str(lastname))
    if lastname == None:
        return Genderizer.detect(firstName=firstname)
    try:
        p = wikipedia.page(firstname + lastname)
        #print p.title
        text_from_wikipedia = p.content
        if len(text_from_wikipedia) > 1000:
            text_from_wikipedia = text_from_wikipedia[0:1000]
        #        print('text from wiki:'+text_from_wikipedia)
        return Genderizer.detect(firstName=firstname, text=text_from_wikipedia)
    except wikipedia.exceptions.PageError:
        return Genderizer.detect(firstName=firstname)
    except wikipedia.exceptions.DisambiguationError as e:
        print e.options
        first_option = e.options[0]
        print first_option
        try:
            p = wikipedia.page(first_option)
            print p.title
            text_from_wikipedia = p.content
            if len(text_from_wikipedia) > 1000:
                text_from_wikipedia = text_from_wikipedia[0:1000]
                #        print('text from wiki:'+text_from_wikipedia)
            return Genderizer.detect(firstName=firstname, text=text_from_wikipedia)
        except wikipedia.exceptions.DisambiguationError as e:
            return Genderizer.detect(firstName=firstname)


if __name__ == '__main__':
    print 'starting'
    #gender = determine_gender_from_wiki('Aaron','Swartz')
    #print gender
    cdir = os.getcwd()
    dir = os.path.join(cdir, 'classifier_stuff/male-female/lfw-deepfunneled')
    dir = '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/classifier_stuff/male-female/lfw-deepfunneled'
    subdirs = Utils.immediate_subdirs(dir)
    for subdir in subdirs:
        dirname = subdir.split('/')[-1]
        names = dirname.split('_')
        firstname = names[0]
        restofname = ''
        for i in range(1, len(names)):
            restofname = restofname + ' ' + names[i]
        gender = determine_gender_from_wiki(firstname, lastname=restofname)
        if gender is None:
            gender = 'None'
        print(firstname + ' ' + restofname + ' is ' + gender)
        if gender is 'male':
            dst = os.path.join(dir, 'male')
            dst = os.path.join(dst, dirname)
            shutil.move(subdir, dst)
        elif gender is 'female':
            dst = os.path.join(dir, 'female')
            dst = os.path.join(dst, dirname)
            shutil.move(subdir, dst)
        else:
            dst = os.path.join(dir, 'unknown')
            dst = os.path.join(dst, dirname)
            shutil.move(subdir, dst)
            #raw_input('enter to continue')


            '''
            CORRECTION TO genderizer.py - push to their git or something:
            from line 83: (to fix division by 0 when sum(prob.values())==0)

                                    if sum(probablities.values()):
                classifierScoreLogF = probablities['female'] / sum(probablities.values())
                classifierScoreLogM = probablities['male'] / sum(probablities.values())
                classifierScoreM = classifierScoreLogF / (classifierScoreLogM + classifierScoreLogF)
                classifierScoreF = classifierScoreLogM / (classifierScoreLogM + classifierScoreLogF)
                if nameGender and nameGender['gender'].startswith('?'):
                    if nameGender['gender'] == cls.mostlyMale and classifierScoreM > 0.6:
                        return 'male'
                    elif nameGender['gender'] == cls.mostlyFemale and classifierScoreF > 0.6:
                        return 'female'
                    elif nameGender['gender'] != cls.genderUnknown:
                        return None

                # If there is no information according to the name and
                # there is significant difference between the two probablity,
                # we can accept the highest probablity.
                if abs(classifierScoreF - classifierScoreM) > cls.significantDegree:
                    if probablities['female'] > probablities['male']:
                        return 'female'
                    else:
                        return 'male'

            else:
                return None
                            '''