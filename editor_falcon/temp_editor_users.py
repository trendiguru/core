from passlib.hash import sha256_crypt


USERS = {
    "john@john.co":
    {
        "email": "john@john.co",
        "password": sha256_crypt.encrypt("john_is_cool")
    },
    "stylebook@stylebook.de":
    {
        "email": "stylebook@stylebook.de",
        "password": sha256_crypt.encrypt("stylebook")
    }
}