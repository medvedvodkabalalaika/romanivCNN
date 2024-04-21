from datetime import datetime
import platform



system = platform.system()

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def statWin(dName,EP, LR, bSize, correct, total, acp, eta):
    filename = f'statistics/statistics-{timestamp}.cmd'
    with open(filename, 'w') as file:
        file.write(f'@echo off\n')
        file.write(f'color 9\n')
        file.write(f'echo Date and time: {timestamp}\n')
        file.write(f'echo Device: {dName}\n')
        file.write('\n')
        file.write(f'echo Number of epochs: {EP}\n')
        file.write(f'echo Learning rate: {LR}\n')
        file.write(f'echo Batch size: {bSize}\n')
        file.write('\n')
        file.write(f'echo Network accuracy: {correct} out of {total} ({acp:.2f}%)\n')
        file.write(f'echo Training time: {eta:.2f} seconds\n')
        file.write('\n')
        file.write(f'echo romanivske/romanivCNN\n')
        file.write('pause')

    return filename

def statMac(dName,EP, LR, bSize, correct, total, acp, eta):
    with open(f'statistics/statistics-{timestamp}.txt', 'w') as file:
        file.write(f'Дата и время: {timestamp}\n')
        file.write(f'Устройство: {dName}\n')
        file.write('\n')
        file.write(f'Количество эпох: {EP}\n')
        file.write(f'Скорость обучения: {LR}\n')
        file.write(f'Размер пакета: {bSize}\n')
        file.write('\n')
        file.write(f'Точность сети: {correct} из {total} ({acp:.2f}%)\n')
        file.write(f'Время обучения: {eta:.2f} секунд\n')
        file.write('\n')
        file.write('romanivske/romanivCNN')