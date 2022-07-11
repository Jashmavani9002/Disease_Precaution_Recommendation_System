from django.shortcuts import render
from requests import request
from rest_framework.decorators import api_view
from rest_framework.response import Response
from training.Utils.train_model import training
from rest_framework import status

# Create your views here.
@api_view(['GET'])
def disease_train(request):
    if request.method == 'GET':
        try:
            training()
            return Response(status=status.HTTP_202_ACCEPTED)
        except Exception as e:
            return Response(status=status.HTTP_400_BAD_REQUEST)