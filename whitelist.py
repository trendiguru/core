__author__ = 'yonatan'
import sys

from rq import Queue

import constants
from crawlme import scrapLinks

scrap_q = Queue('CrawlMe', connection=constants.redis_conn)
db = constants.db

fullList = ["yahoo.com", "msn.com", "yahoo.co.jp", "qq.com", "uol.com.br", "globo.com", "naver.com", "onet.pl",
             "espn.go.com", "news.yahoo.com", "163.com", "wp.pl", "sina.com.cn", "news.google.com", "bbc.co.uk",
             "cnn.com", "news.yandex.ru", "rambler.ru", "bbc.com", "cnet.com", "dailymail.co.uk", "milliyet.com.tr",
             "nytimes.com", "news.mail.ru", "zing.vn", "sports.yahoo.com", "news.yahoo.co.jp", "theguardian.com",
             "buzzfeed.com", "interia.pl", "indiatimes.com", "hurriyet.com.tr", "huffingtonpost.com", "ifeng.com",
             "t-online.de", "foxnews.com", "drudgereport.com", "sohu.com", "about.com", "weather.com", "rediff.com",
             "iqiyi.com", "bp.blogspot.com", "livedoor.jp", "repubblica.it", "wunderground.com", "vnexpress.net",
             "forbes.com", "lemonde.fr", "bloomberg.com", "mynet.com", "telegraph.co.uk", "naver.jp", "ukr.net",
             "o2.pl", "idnes.cz", "24h.com.vn", "usatoday.com", "ig.com.br", "news.163.com", "washingtonpost.com",
             "spiegel.de", "gazeta.pl", "rt.com", "gismeteo.ru", "elpais.com", "marca.com", "pixnet.net",
             "accuweather.com", "rbc.ru", "sapo.pt", "haberturk.com", "news.rambler.ru", "goo.ne.jp",
             "commentcamarche.net", "libero.it", "sozcu.com.tr", "espncricinfo.com", "clarin.com", "liveleak.com",
             "corriere.it", "news.ifeng.com", "bol.uol.com.br", "bild.de", "pantip.com", "lequipe.fr", "ynet.co.il",
             "r7.com", "lanacion.com.ar", "folha.uol.com.br", "ccm.net", "hupu.com", "elmundo.es", "rojadirecta.me",
             "bleacherreport.com", "vice.com", "news.qq.com", "sabah.com.tr", "news.sina.com.cn", "infobae.com",
             "Domain", "avon.ru", "wizaz.pl", "women.kapook.com", "minq.com", "sephora.fr", "birchbox.com",
             "totalbeauty.com", "fragrantica.com", "fragrancenet.com", "kiehls.com", "ladyfare.com", "nocibe.fr",
             "yves-rocher.fr", "lushusa.com", "beaute-test.com", "herbeauty.co", "ipsy.com", "fragrancex.com",
             "eva.ro", "fashionguide.com.tw", "byrdie.com", "jeban.com", "esteelauder.com", "allbeauty.com",
             "stitchfix.com", "shiseido.co.jp", "zdface.com", "origins.com", "lrworld.com", "skincare-univ.com",
             "primlifestyles.com", "telva.com", "nyxcosmetics.com", "thebodyshop.co.uk", "staggeringbeauty.com",
             "stylist.co.uk", "xovain.com", "purplle.com", "urcosme.com", "tartecosmetics.com", "cosme-de.com",
             "ambitiousbeauty.com", "kao.com", "prestigeladies.com", "birchbox.fr", "essence.eu", "kimiss.com",
             "vdolady.com", "coastalscents.com", "lifestylebite.com", "wahanda.com", "locatel.com.ve", "rasysa.com",
             "docebeleza.com.br", "narscosmetics.com", "hermo.my", "thebeautydepartment.com", "dailymakeover.com",
             "loreal-paris.fr", "nouveaucheap.blogspot.com", "crabtree-evelyn.com", "parfumo.de", "shiseido.com",
             "cultbeauty.co.uk", "clarinsusa.com", "youbeauty.com", "musingsofamuse.com", "laurenconrad.com",
             "shuuemura-usa.com", "escentual.com", "thebeautybox.com.br", "shopunt.com", "yslbeautyus.com",
             "elizabetharden.com", "lush.com", "blivakker.no", "thebeautyst.com", "ghdhair.com", "vichyconsult.ru",
             "urodaizdrowie.pl", "nuxe.com", "ofracosmetics.com", "rimmellondon.com", "elle.nl", "stylesatlife.com",
             "loreal-paris.es", "maccosmetics.fr", "perfume.com", "newbeauty.com", "allcosmeticswholesale.com",
             "luckyscent.com", "opi.com", "b-glowing.com", "fragrantica.es", "sundaymore.com", "jomalone.com",
             "diva.by", "ediva.gr", "bgo.com.tw", "glossybox.com", "pudelek.pl", "tmz.com", "super.cz", "ew.com",
             "entretenimento.r7.com", "hollywoodlife.com", "kapanlagi.com", "zimbio.com", "jezebel.com",
             "purepeople.com", "jeanmarcmorandini.com", "radaronline.com", "etonline.com", "voici.fr", "topito.com",
             "ciudad.com.ar", "perezhilton.com", "virgula.uol.com.br", "suggest.com", "justjared.com", "therichest.com",
             "pressroomvip.com", "dagospia.com", "closermag.fr", "pagesix.com", "digitalspy.com", "purepeople.com.br",
             "thepiratebay.uk.net", "sopitas.com", "deadline.com", "starpulse.com", "primiciasya.com", "celebuzz.com",
             "luckstars.co", "ratingcero.com", "non-stop-people.com", "tochka.net", "toofab.com", "extra.cz",
             "kozaczek.pl", "huabian.com", "bossip.com", "spletnik.ru", "wetpaint.com", "promiflash.de", "gala.fr",
             "natalie.mu", "public.fr", "digitalspy.co.uk", "bollywoodlife.com", "tiramillas.net", "storm.mg",
             "thesuperficial.com", "2sao.vn", "oprah.com", "hecklerspray.com", "gossiplankanews.com",
             "guiadasemana.com.br", "hiddenplaybook.com", "celebritynetworth.com", "vietgiaitri.com", "gossip-tv.gr",
             "okmagazine.com", "ohtuleht.ee", "alternatifim.com", "egokick.com", "relax.by", "kanyetothe.com",
             "coed.com", "tvnotas.com.mx", "lifeandstylemag.com", "hawtcelebs.com", "lazygirls.info", "sabay.com.kh",
             "hypescience.com", "wwtdd.com", "pinkvilla.com", "fusion.net", "betterfap.com", "grabo.bg",
             "madamenoire.com", "hitfix.com", "ahaonline.cz", "123telugu.com", "nofilmschool.com", "dongtw.com",
             "playgroundmag.net", "ngoisao.vn", "potins.net", "wowkeren.com", "movistar-adsl.com", "purebreak.com",
             "entertaintastic.com", "dioguinho.pt", "perfecte.ro", "napi.hu", "klubas.lt", "esmas.com", "vertele.com",
             "purebreak.com.br", "maxmodels.pl", "stylebistro.com", "fashion.ifeng.com", "tajbao.com",
             "highsnobiety.com", "osinka.ru", "trendsylvania.net", "whowhatwear.com", "fashiony.ru",
             "fashion.sina.com.cn", "lookbook.nu", "vogue.com.tw", "thefashionspot.com", "elle.com.tw",
             "vogue.com.cn", "thehunt.com", "fashionbeans.com", "models.com", "fashion.sohu.com", "purseblog.com",
             "elle.co.jp", "perfecte.md", "cosmopolitan.lt", "wwd.com", "basenotes.net", "wmagazine.com", "enrz.com",
             "moteris.lt", "styleblazer.com", "theblondesalad.com", "fashiongonerogue.com", "thesartorialist.com",
             "fashionnstyle.com", "fashion.walla.co.il", "thegloss.com", "vogue.com.au", "panele.lt",
             "collegefashion.net", "niv.ru", "desired.de", "fashionstylemag.com", "guimi.com", "fashionbank.ru",
             "vmagazine.com", "garancedore.fr", "thefashionisto.com", "fashionising.com", "modelmanagement.com",
             "mrvintage.pl", "picbazi.com", "self.com.cn", "marko.by", "lovemaegan.com", "fashiontime.ru",
             "harpersbazaar.com.hk", "stylight.com", "dietbook.biz", "harpersbazaar.co.uk", "mybodygallery.com",
             "blyzka.by", "she.com", "thecoveteur.com", "rewardstyle.com", "fustany.com", "stylishlyme.com",
             "fashion-headline.com", "9linesmag.com", "vakko.com", "fashion.telegraph.co.uk", "fashionbombdaily.com",
             "instyle.de", "stylebible.ph", "graziadaily.co.uk", "fimela.com", "brandcircus.ro", "mkstyle.net",
             "purestorm.com", "fashiontv.com", "fashionmagazine.com", "meltyfashion.fr", "avili-style.ru",
             "tech.xinmin.cn", "followme.gr", "popbee.com", "celebsvenue.com", "fashion.hola.com", "todaysfashion.com",
             "fashion.allwomenstalk.com", "hellomollyfashion.com", "magazine-data.com", "style.com",
             "wearesodroee.com", "eazyfashion.com", "candidfashionpolice.com", "malemodelscene.net", "starttoday.jp",
             "nifmagazine.com", "harpersbazaar.com.au", "levi.com.cn", "femlife.de", "tmz.com", "ew.com", "jezebel.com",
             "hollywoodlife.com", "etonline.com", "radaronline.com", "zimbio.com", "perezhilton.com", "pagesix.com",
             "suggest.com", "starpulse.com", "therichest.com", "bossip.com", "deadline.com", "toofab.com", "oprah.com",
             "hiddenplaybook.com", "pressroomvip.com", "egokick.com", "okmagazine.com", "justjared.com", "celebuzz.com",
             "wetpaint.com", "thesuperficial.com", "entertaintastic.com", "celebritynetworth.com", "madamenoire.com",
             "wwtdd.com", "buzzlamp.com", "fusion.net", "liftbump.com", "lifeandstylemag.com", "kanyetothe.com",
             "coed.com", "fishwrapper.com", "hitfix.com", "hecklerspray.com", "playbill.com", "denofgeek.us",
             "betterfap.com", "insideedition.com", "gofugyourself.com", "nofilmschool.com", "hellobeautiful.com",
             "theawl.com", "digitalspy.com", "soapcentral.com", "lazygirls.info", "wonderwall.com",
             "thesmokinggun.com", "screencrush.com", "gossipcop.com", "celebdirtylaundry.com", "the-toast.net",
             "geektyrant.com", "celebrityfanatic.com", "realitytea.com", "blindgossip.com", "celebitchy.com",
             "laineygossip.com", "maxgo.com", "pinkvilla.com", "fame10.com", "tomandlorenzo.com",
             "crazydaysandnights.net", "rumorfix.com", "justjaredjr.com", "bustedcoverage.com",
             "rumordaily.com", "comicsalliance.com", "splitsider.com", "accesshollywood.com",
             "celebritypix.com", "hollywoodtuna.com", "movieandtvcorner.com", "extratv.com",
             "facade.com", "popoholic.com", "rantchic.com", "starcasm.net", "sbux-portal.appspot.com",
             "thehollywoodgossip.com", "redlettermedia.com", "hawtcelebs.com", "closerweekly.com", "dvdtalk.com",
             "hypable.com", "soapsindepth.com", "mediamass.net", "theforce.net", "famefocus.com", "storm.mg",
             "accessthestars.com", "bollywoodlife.com", "screenjunkies.com", "starmagazine.com", "digitalspy.co.uk",
             "dayscafe.com", "barevhayer.com", "collegecandy.com", "stylebistro.com", "whowhatwear.com",
             "highsnobiety.com", "thehunt.com", "purseblog.com", "thefashionspot.com", "lookbook.nu", "wwd.com",
             "fashionbeans.com", "models.com", "wmagazine.com", "basenotes.net", "thegloss.com", "collegefashion.net",
             "fashionnstyle.com", "styleblazer.com", "mybodygallery.com", "stylight.com", "picbazi.com",
             "thesartorialist.com", "stylishlyme.com", "vogue.com.tw", "fashiongonerogue.com", "thecoveteur.com",
             "fashion.ifeng.com", "rewardstyle.com", "scoopexchange.com", "vmagazine.com", "todaysfashion.com",
             "garancedore.fr", "theblondesalad.com", "candidfashionpolice.com", "thefashionisto.com",
             "fashion.sina.com.cn", "elle.com.tw", "fashion.allwomenstalk.com", "iahfy.tumblr.com",
             "acontinuouslean.com", "fashionscope.com", "report-site.com", "modelmanagement.com",
             "fashionbombdaily.com", "stylesoftomorrow.com", "fashionising.com", "vogue.com.au",
             "fashion.telegraph.co.uk", "adriannapapell.com", "stylemined.com", "fashionstylemag.com",
             "harpersbazaar.co.uk", "lovemaegan.com", "getkempt.com", "fashion.net", "vogue.com.cn",
             "wallisfashion.com", "osinka.ru", "fashion.about.com", "wantable.com", "fimela.com", "fashion-era.com",
             "hellomollyfashion.com", "celebsvenue.com", "perfecte.md", "fashionmagazine.com", "songofstyle.com",
             "insideoutstyleblog.com", "hespokestyle.com", "styleite.com", "galadarling.com", "whatiwore.tumblr.com",
             "style.com", "kpophqpictures.co.vu", "fashionnewbie.com", "maxmodels.pl", "lcking.com",
             "lovethepinups.tumblr.com", "newsofday.com", "weddingforward.com", "thebudgetfashionista.com",
             "glamourdaze.com", "wardrobeoxygen.com", "fazmeutipo.tumblr.com", "marcel-capato.com",
             "tokyofashion.com", "fashionfixation.com", "fabulousafter40.com", "fashionisers.com",
             "frappuccini.tumblr.com", "self.com.cn", "fashionweekdaily.com", "doyoulikevintage.tumblr.com",
             "cs10.org", "nitrolicious.com", "vickumbro.tumblr.com", "malemodelscene.net", "ironandtweed.com",
             "fashiontoast.com", "flare.com"]

fashionOnly = ["manrepeller.com", "wishwishwish.net", "parkandcube.com", "stellaswardrobe.com", "cocosteaparty.com",
               "5inchandup.blogspot.co.uk", "garypeppergirl.com", "camilleovertherainbow.com", "streetpeeper.com",
               "the-frugality.com", "disneyrollergirl.net", "weworewhat.com", "wearingittoday.co.uk",
               "ella-lapetiteanglaise.com",
               "advancedstyle.blogspot.co.uk", "indtl.com", "redcarpet-fashionawards.com", "nadiaaboulhosn.com",
               "enbrogue.com",
               "peonylim.com", "vanessajackman.blogspot.co.uk", "alltheprettybirds.com", "lisegrendene.com.br",
               "nataliehartleywears.blogspot.co.uk", "tommyton.com", "stylebubble.co.uk", "pandorasykes.com",
               "theblondesalad.com",
               "thesartorialist.com", "bryanboy.com", "bunte.de", "gala.fr"]


def masterCrawler(floor=2, whiteList=fashionOnly):
    db.crawler_processed.drop()
    db.crawler_processed.create_index("url")
    for site in whiteList:
        url = "http://www." + site
        scrap_q.enqueue(scrapLinks, url, floor)
    return "finished"


if __name__ == "__main__":
    print ("Scraping the white list - Started...)")
    floor = 2
    if len(sys.argv) == 2:
        floor = int(sys.argv[1])
    res = masterCrawler(floor)
    print (res)
