import tldextract

from . import constants

db = constants.db


def add_to_whitelist(url):
    try:
        reg = tldextract.extract(url).registered_domain
        if not db.whitelist.find_one({'domain': reg}):
            db.whitelist.insert_one({'domain': reg})
            return "Done! :)"
        else:
            return reg + " is already in the WhiteList! :)"
    except Exception as e:
        return e