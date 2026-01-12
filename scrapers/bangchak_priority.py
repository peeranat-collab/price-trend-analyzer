from scrapers.bangchak_diesel_advanced import get_monthly_average_advanced
from scrapers.bangchak_diesel_basic import get_monthly_average_basic

def get_diesel_price_with_priority(year, month):
    # 1) Try Advanced
    try:
        return get_monthly_average_advanced(year, month)
    except Exception as adv_err:
        advanced_error = str(adv_err)

    # 2) Try Basic
    try:
        return get_monthly_average_basic(year, month)
    except Exception as basic_err:
        basic_error = str(basic_err)

    # 3) Fallback
    return {
        "status": "fallback",
        "reason": f"Advanced: {advanced_error} | Basic: {basic_error}"
    }
