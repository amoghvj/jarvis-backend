from app.services.github_oauth import pick_verified_email


def test_prefers_primary_verified():
    emails = [
        {"email": "a@x.com", "verified": True, "primary": False},
        {"email": "b@x.com", "verified": True, "primary": True},
        {"email": "c@x.com", "verified": False, "primary": True},
    ]
    assert pick_verified_email(emails) == "b@x.com"


def test_ignores_unverified():
    assert pick_verified_email([{"email": "c@x.com", "verified": False, "primary": True}]) is None


def test_falls_back_to_any_verified():
    assert pick_verified_email([{"email": "a@x.com", "verified": True, "primary": False}]) == "a@x.com"

