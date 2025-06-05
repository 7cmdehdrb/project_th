cmake_minimum_required(VERSION 3.5)
project(object_tracker)

# 필수 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(Python3 REQUIRED)

# install.py를 설치 디렉토리에 복사
install(
  PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/object_tracker/install.py
  DESTINATION lib/${PROJECT_NAME}
)

# colcon install 후 install.py 자동 실행
install(
  CODE "execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/install.py)"
)

ament_package()
