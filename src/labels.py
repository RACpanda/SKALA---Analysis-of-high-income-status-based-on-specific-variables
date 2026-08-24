"""사용자 화면과 시각화에서 공통으로 사용하는 표시용 라벨.

원본 데이터의 컬럼명·범주값은 분석 및 모델 입력에서 그대로 유지하고,
이 모듈은 화면에 보여줄 한국어 표현만 관리한다.
"""

from __future__ import annotations


VARIABLE_LABELS = {
    "age": "나이",
    "workclass": "고용 형태",
    "education": "교육 수준",
    "marital-status": "혼인 상태",
    "occupation": "직업",
    "relationship": "가구 내 관계",
    "race": "인종",
    "sex": "성별",
    "capital-gain": "투자·자산 이익",
    "capital-loss": "투자·자산 손실",
    "hours-per-week": "주당 근무시간",
    "native-country": "출신 국가",
}


CATEGORY_VALUE_LABELS = {
    "sex": {
        "Male": "남성",
        "Female": "여성",
    },

    "race": {
        "White": "백인",
        "Black": "흑인",
        "Asian-Pac-Islander": "아시아·태평양계",
        "Amer-Indian-Eskimo": "아메리카 원주민·알래스카 원주민",
        "Other": "기타",
    },

    "relationship": {
        "Husband": "남편",
        "Wife": "아내",
        "Own-child": "자녀",
        "Not-in-family": "가족 외",
        "Other-relative": "기타 친족",
        "Unmarried": "미혼·비혼",
    },

    "marital-status": {
        "Married-civ-spouse": "기혼·배우자 동거",
        "Divorced": "이혼",
        "Never-married": "미혼",
        "Separated": "별거",
        "Widowed": "사별",
        "Married-spouse-absent": "기혼·배우자 부재",
        "Married-AF-spouse": "군인 배우자와 기혼",
    },

    "workclass": {
        "Private": "민간 기업",
        "Self-emp-not-inc": "자영업·비법인",
        "Self-emp-inc": "자영업·법인",
        "Federal-gov": "연방정부",
        "Local-gov": "지방정부",
        "State-gov": "주정부",
        "Without-pay": "무급 근무",
        "Never-worked": "근무 경험 없음",
    },

    "education": {
        "Preschool": "취학 전",
        "1st-4th": "초등 1~4학년",
        "5th-6th": "초등 5~6학년",
        "7th-8th": "중학교 수준",
        "9th": "9학년",
        "10th": "10학년",
        "11th": "11학년",
        "12th": "12학년",
        "HS-grad": "고등학교 졸업",
        "Some-college": "대학 일부 이수",
        "Assoc-voc": "전문학사·직업 과정",
        "Assoc-acdm": "전문학사·학술 과정",
        "Bachelors": "학사",
        "Masters": "석사",
        "Prof-school": "전문대학원",
        "Doctorate": "박사",
    },

    "occupation": {
        "Adm-clerical": "사무·행정직",
        "Armed-Forces": "군인",
        "Craft-repair": "기능·수리직",
        "Exec-managerial": "관리·경영직",
        "Farming-fishing": "농림·어업",
        "Handlers-cleaners": "운반·청소직",
        "Machine-op-inspct": "기계 조작·검사직",
        "Other-service": "기타 서비스직",
        "Priv-house-serv": "가사 서비스직",
        "Prof-specialty": "전문직",
        "Protective-serv": "보안·보호 서비스직",
        "Sales": "판매직",
        "Tech-support": "기술 지원직",
        "Transport-moving": "운송직",
    },

    "native-country": {
        "United-States": "미국",
        "Canada": "캐나다",
        "Mexico": "멕시코",
        "Puerto-Rico": "푸에르토리코",
        "Cuba": "쿠바",
        "Jamaica": "자메이카",
        "Dominican-Republic": "도미니카공화국",
        "Haiti": "아이티",
        "Guatemala": "과테말라",
        "Honduras": "온두라스",
        "Nicaragua": "니카라과",
        "El-Salvador": "엘살바도르",
        "Trinadad&Tobago": "트리니다드 토바고",

        "England": "영국",
        "Germany": "독일",
        "France": "프랑스",
        "Italy": "이탈리아",
        "Poland": "폴란드",
        "Portugal": "포르투갈",
        "Ireland": "아일랜드",
        "Greece": "그리스",
        "Hungary": "헝가리",
        "Scotland": "스코틀랜드",
        "Yugoslavia": "유고슬라비아",
        "Holand-Netherlands": "네덜란드",

        "India": "인도",
        "China": "중국",
        "Japan": "일본",
        "Vietnam": "베트남",
        "Philippines": "필리핀",
        "Thailand": "태국",
        "Cambodia": "캄보디아",
        "Laos": "라오스",
        "Taiwan": "대만",
        "Hong": "홍콩",

        "Iran": "이란",

        "Columbia": "콜롬비아",
        "Ecuador": "에콰도르",
        "Peru": "페루",

        # Adult 데이터의 원문 값만으로 특정 국가를 확정하기 어려워
        # 임의 번역하지 않고 원문 표기를 유지한다.
        "South": "South",

        "Outlying-US(Guam-USVI-etc)": "미국령 지역",
    },
}


VARIABLE_TYPE_LABELS = {
    "binary": "이진형",
    "continuous": "연속형",
    "categorical": "범주형",
}