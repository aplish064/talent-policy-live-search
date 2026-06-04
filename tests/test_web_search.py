from __future__ import annotations

from talent_policy_search.web_search import parse_anysearch_results


def test_parse_anysearch_results_reads_v1_response_shape():
    payload = {
        "code": 0,
        "message": "success",
        "data": {
            "results": [
                {
                    "title": "人才政策-广州市人力资源和社会保障局网站",
                    "url": "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/",
                    "content": "广州人才政策",
                },
                {
                    "title": "duplicate",
                    "url": "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/",
                    "content": "duplicate",
                },
                {
                    "title": "黄埔人才津贴",
                    "url": "http://www.hp.gov.cn/post_10358329.html",
                    "description": "人才津贴",
                },
            ]
        },
    }

    results = parse_anysearch_results(payload, limit=5)

    assert [result.url for result in results] == [
        "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/",
        "http://www.hp.gov.cn/post_10358329.html",
    ]
    assert results[0].title == "人才政策-广州市人力资源和社会保障局网站"
    assert results[0].snippet == "广州人才政策"
    assert results[0].source == "anysearch"


def test_parse_anysearch_results_ignores_malformed_items_and_honors_limit():
    payload = {
        "results": [
            "bad",
            {"title": "missing url"},
            {"url": "https://www.gz.gov.cn/a.html"},
            {"url": "https://www.gz.gov.cn/b.html"},
        ]
    }

    results = parse_anysearch_results(payload, limit=1)

    assert len(results) == 1
    assert results[0].title == "https://www.gz.gov.cn/a.html"
