import math
import random


class Solution:
    def getMinDistSum(self, positions):
        # 温度变化率
        K = 0.97
        # 结束温度，控制降温时间
        EPS = 1E-15

        # 计算自变量的增量
        def get(T):
            return T * random.randint(-32767, 32767)

        def calc(positions, nowx, nowy):
            ans = 0
            for x, y in positions:
                ans += (abs(nowx - x) ** 2 + abs(nowy - y) ** 2) ** 0.5
            return ans

        x0, y0 = 0, 0
        for x, y in positions:
            x0 += x
            y0 += y
        x0 /= len(positions)
        y0 /= len(positions)
        ans = calc(positions, x0, y0)
        cnt = 2
        print("首次数据：距离:{} {} {}".format(ans, x, y))
        while cnt > 0:
            cnt -= 1
            cur = ans
            x1, y1 = x0, y0
            T = 100000
            # 初始温度
            while T > EPS:
                x2, y2 = x1 + get(T), y1 + get(T)
                temp = calc(positions, x2, y2)

                print("当前数据：距离:{}  {},   {},   {}".format(T,temp, x2, y2))
                if temp < ans:
                    ans = temp
                    x0, y0 = x2, y2
                if cur > temp or math.exp((cur - temp) / T) > random.random():
                    cur = temp
                    x1, y1 = x2, y2
                T *= K
        print("当前位置:",x0,y0)
        return ans


if __name__ == '__main__':
    # pos = [[1,1],[3,3]]
    pos = [[1, 1], [0, 0], [2, 0]]
    a = Solution()
    ans = a.getMinDistSum(pos)
    print(ans)
