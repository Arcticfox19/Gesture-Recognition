import pyautogui
import time
class ControlBilibili:
    def __init__(self) -> None:
        pass
    @staticmethod
    def stopAndPlayVideo() -> bool:  
        """
        暂停/播放
        :param position:
        :return:
        """
        print("播放/暂停")
        pyautogui.hotkey(" ")
        time.sleep(1)
    @staticmethod
    def nextVideo() -> bool:
        """
        下一个视频
        :param position:
        :return:
        """
        print("下一个视频")
        pyautogui.hotkey("]")
    @staticmethod
    def lastVideo() -> bool:
        """
        上一个视频
        :param position:
        :return:
        """
        print("上一个视频")
        pyautogui.hotkey("[")
    @staticmethod
    def speedUp() -> bool:
        """
        快进
        :param position:
        :return:
        """
        print("快进")
        pyautogui.hotkey("right")
    @staticmethod
    def speedDown() -> bool:
        """
        快退
        :param position:
        :return:
        """
        print("快退")
        pyautogui.hotkey("left")
    @staticmethod
    def mute() -> bool:
        """
        静音/音量
        :param position:
        :return:
        """
        print("静音")
        pyautogui.hotkey("m")
        time.sleep(1)
    @staticmethod
    def praise() -> bool:
        """
        点赞
        :param position:
        :return:
        """
        print("点赞")
        pyautogui.hotkey("q")

    @staticmethod
    def threeEven() -> bool:
        """
        一键三连
        :param position:
        :return:
        """
        print("一键三连")
        pyautogui.keyDown("q")
        time.sleep(3)
        pyautogui.keyUp("q")

    @staticmethod
    def FullScreen() -> bool:
        """
        全屏
        :param position:
        :return:
        """
        print("全屏")
        pyautogui.hotkey("f")
if __name__=="__main__":
    print("准备暂停或者播放完成")
    time.sleep(5)
    ControlBilibili.pauseBreak()
    print("暂停或者播放完成")