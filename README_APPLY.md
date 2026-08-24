# v1.4 문의하기 기능 적용

포함 파일:
- `app.py`: 문의 UI 통합본
- `src/inquiry.py`: 문의 검증·레코드 생성·로컬 CSV 저장
- `tests/test_inquiry.py`: 문의 기능 단위 테스트

현재 로컬 저장 위치는 `outputs/inquiries/user_inquiries.csv`입니다. 이 CSV 방식은 로컬 개발/검증용입니다. 웹 배포 환경에서 로컬 파일 영속성이 보장되지 않으면 `save_user_inquiry()` 저장 구현을 DB/API 등 영속 저장소로 교체해야 합니다.

적용 전 현재 `app.py`를 백업한 뒤 파일을 복사하고 다음을 실행합니다.

```bash
python -m pytest -q -W error
python -m streamlit run app.py
```

문의 제출 후 CSV에 `inquiry_id`, `created_at`, `version=1.4`, `current_page`, `category`, `message`, `email`, `status=접수`가 저장되는지 확인합니다.
