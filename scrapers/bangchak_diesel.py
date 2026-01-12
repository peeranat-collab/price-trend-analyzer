from scrapers.bangchak_diesel_advanced import get_monthly_average_advanced, BangchakAdvancedScraperError
from scrapers.bangchak_diesel import get_monthly_average as get_basic

def get_diesel_price_with_priority(year, month):
    # 1) Try Advanced
    try:
        adv = get_monthly_average_advanced(year, month)
        return {
            "status": "advanced",
            "value": adv
        }
    except Exception as e:
        adv_error = str(e)

    # 2) Try Basic
    basic = get_basic(year, month)
    if not isinstance(basic, dict):
        return {
            "status": "basic",
            "value": basic
        }

    # 3) Manual fallback
    return {
        "status": "manual",
        "reason": adv_error
    }
