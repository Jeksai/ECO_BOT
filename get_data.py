import asyncio
import aiohttp
import pandas as pd
from data.TOKEN import API_KEY

URL = "https://apidata.mos.ru/v1/datasets/2453/rows"
TOP = 1000
BATCH_SIZE = 10


async def fetch_page(session: aiohttp.ClientSession, skip: int) -> list:
    params = {"api_key": API_KEY, "$top": TOP, "$skip": skip}
    async with session.get(URL, params=params) as response:
        if response.status != 200:
            print(f"Ошибка запроса (status {response.status}) для $skip={skip}")
            return []
        data = await response.json()
        return data


async def main():
    all_data = []
    async with aiohttp.ClientSession() as session:
        page_index = 0
        while True:
            tasks = []
            for i in range(BATCH_SIZE):
                skip = (page_index + i) * TOP
                tasks.append(fetch_page(session, skip))
            print(
                f"Запрашиваем страницы с {(page_index)*TOP} до {((page_index+BATCH_SIZE-1)*TOP)}"
            )
            results = await asyncio.gather(*tasks)

            stop = False
            for res in results:
                if not res:
                    stop = True
                    break
                all_data.extend(res)
            if stop:
                print("Достигнут конец данных.")
                break
            page_index += BATCH_SIZE

    df = pd.DataFrame(all_data)

    if "Cells" in df.columns:
        df = pd.json_normalize(df["Cells"])

    print(df.info())

    df.to_csv("data/data.csv", sep=";", index=False)


if __name__ == "__main__":
    asyncio.run(main())
